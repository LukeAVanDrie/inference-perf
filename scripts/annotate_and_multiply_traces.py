#!/usr/bin/env python3
# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OTel Trace Annotator & Multiplier utility.

This script reads a baseline OTel trace, replicates it into multiple sessions,
regenerates Trace and Span IDs to prevent collisions while maintaining DAG structure,
and annotates them with multi-tenant labels based on a configured distribution.
"""

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
from typing import Any, Dict, List, cast


def positive_int(value: str) -> int:
    """Validates that the value is a positive integer >= 1."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: '{value}'") from None
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {ivalue}")
    return ivalue


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description="OTel Trace Annotator & Multiplier")
    parser.add_argument("--baseline-file", required=True, help="Path to valid baseline OTel trace JSON file")
    parser.add_argument("--output-dir", required=True, help="Target directory to write generated trace files")
    parser.add_argument(
        "--num-sessions", type=positive_int, required=True, help="Total number of unique trace files to generate"
    )
    parser.add_argument("--labels-config", required=True, help="Path to JSON config file OR inline stringified JSON block")
    parser.add_argument(
        "--clean", action="store_true", default=False, help="Clean the output directory before generating new trace files"
    )
    return parser.parse_args()


def load_config(config_input: str) -> Dict[str, Any]:
    """Loads config from inline JSON string or file path."""
    stripped_input = config_input.strip()
    if stripped_input.startswith("{") or stripped_input.startswith("["):
        try:
            data = json.loads(stripped_input)
            if not isinstance(data, dict):
                raise ValueError("Configuration must be a JSON object")
            return cast(Dict[str, Any], data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse inline JSON config: {e}") from e
    else:
        try:
            with open(config_input, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("Configuration must be a JSON object")
                return cast(Dict[str, Any], data)
        except Exception as e:
            raise ValueError(f"Failed to load labels config from path '{config_input}': {e}") from e


def normalize_weights(items: List[Dict[str, Any]], weight_key: str = "weight") -> None:
    """Normalizes weights in-place to sum up exactly to 1.0."""
    total_weight = sum(item.get(weight_key, 0.0) for item in items)
    if total_weight == 0.0:
        raise ValueError("Total weight of distribution cannot be zero.")

    if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
        for item in items:
            item[weight_key] = item.get(weight_key, 0.0) / total_weight


def parse_and_normalize_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parses the dual-mode configuration and normalizes weights."""
    mode = config.get("mode")
    if not mode:
        raise ValueError("Config is missing required 'mode' key.")

    processed_slices: List[Dict[str, Any]] = []

    if mode == "independent":
        labels = config.get("labels")
        if not labels:
            raise ValueError("Independent mode requires 'labels' key.")

        # Normalize individual label weights first
        normalized_labels: Dict[str, Dict[str, float]] = {}
        for label_key, val_weights in labels.items():
            if not val_weights:
                raise ValueError(f"Weights dictionary for label '{label_key}' is empty.")

            total_w = sum(val_weights.values())
            if total_w == 0.0:
                raise ValueError(f"Total weight for label '{label_key}' cannot be zero.")

            normalized_labels[label_key] = {k: w / total_w for k, w in val_weights.items()}

        # Compute Cartesian product of all labels
        label_keys = list(normalized_labels.keys())
        label_values_lists = [list(normalized_labels[k].keys()) for k in label_keys]

        for combo in itertools.product(*label_values_lists):
            slice_labels = {}
            joint_weight = 1.0
            for k, val in zip(label_keys, combo, strict=True):
                slice_labels[k] = val
                joint_weight *= normalized_labels[k][val]

            processed_slices.append({"labels": slice_labels, "weight": joint_weight})

    elif mode == "joint":
        slices = config.get("slices")
        if not slices:
            raise ValueError("Joint mode requires 'slices' key.")

        # Deep copy to avoid modifying original config in-place
        processed_slices = copy.deepcopy(slices)
        normalize_weights(processed_slices, weight_key="weight")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'independent' or 'joint'.")

    return processed_slices


def distribute_sessions(slices: List[Dict[str, Any]], num_sessions: int) -> List[int]:
    """Distributes num_sessions among slices using the Largest Remainder Method (Hamilton method)."""
    exact_counts = [num_sessions * s.get("weight", 0.0) for s in slices]
    floor_counts = [int(math.floor(x)) for x in exact_counts]
    remainders = [exact - floor for exact, floor in zip(exact_counts, floor_counts, strict=True)]

    missing = num_sessions - sum(floor_counts)

    # Sort indices by remainder in descending order
    indexed_remainders = list(enumerate(remainders))
    indexed_remainders.sort(key=lambda x: x[1], reverse=True)

    counts = list(floor_counts)
    for i in range(missing):
        idx = indexed_remainders[i][0]
        counts[idx] += 1

    return counts


def main() -> int:
    """Main entrypoint."""
    try:
        args = parse_args()
    except Exception as e:
        print(f"CLI argument parsing error: {e}", file=sys.stderr)
        return 1

    try:
        config = load_config(args.labels_config)
        slices = parse_and_normalize_config(config)
    except Exception as e:
        print(f"Error parsing or validating configuration: {e}", file=sys.stderr)
        return 1

    if not os.path.exists(args.baseline_file):
        print(f"Error: Baseline file not found: {args.baseline_file}", file=sys.stderr)
        return 1

    try:
        with open(args.baseline_file, "r", encoding="utf-8") as f:
            baseline = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load baseline file: {e}", file=sys.stderr)
        return 1

    # Identify baseline format: list of spans directly vs. wrapped dict
    if isinstance(baseline, list):
        baseline_spans = baseline
        is_array_format = True
    elif isinstance(baseline, dict) and "spans" in baseline:
        baseline_spans = baseline["spans"]
        is_array_format = False
    else:
        print("Error: Baseline format must be a JSON list of spans or a JSON dict containing 'spans'.", file=sys.stderr)
        return 1

    if not baseline_spans:
        print("Error: Baseline trace contains no spans.", file=sys.stderr)
        return 1

    # Clean output directory if requested
    if args.clean and os.path.exists(args.output_dir):
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Warning: Failed to delete {item_path}: {e}", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)

    session_counts = distribute_sessions(slices, args.num_sessions)

    # Warn if session count is low and some slices receive 0 sessions
    if args.num_sessions < len(slices):
        omitted_slices = []
        for idx, count in enumerate(session_counts):
            if count == 0:
                slice_info = slices[idx]
                labels = slice_info["labels"]
                label_str = ", ".join(f"{k}={labels[k]}" for k in sorted(labels.keys()))
                omitted_slices.append(f"  - {label_str}")

        if omitted_slices:
            print(
                f"WARNING: --num-sessions ({args.num_sessions}) is less than the number of configured slices ({len(slices)}). "
                f"The following slices will receive 0 sessions:",
                file=sys.stderr,
            )
            for slice_str in omitted_slices:
                print(slice_str, file=sys.stderr)

    session_idx = 0
    for slice_idx, count in enumerate(session_counts):
        slice_info = slices[slice_idx]
        labels = slice_info["labels"]

        for _ in range(count):
            # Generate new unique trace_id (32 hex chars)
            new_trace_id = secrets.token_hex(16)

            # Map old span_id to new unique span_id (16 hex chars)
            span_id_map = {}
            for span in baseline_spans:
                old_span_id = span.get("span_id")
                if old_span_id:
                    span_id_map[old_span_id] = secrets.token_hex(8)

            new_spans = []
            for span in baseline_spans:
                new_span = copy.deepcopy(span)

                # Replace trace_id
                new_span["trace_id"] = new_trace_id

                # Replace span_id
                old_span_id = span.get("span_id")
                if old_span_id in span_id_map:
                    new_span["span_id"] = span_id_map[old_span_id]

                # Replace parent_span_id if present in span structure
                if "parent_span_id" in span:
                    old_parent_id = span["parent_span_id"]
                    if old_parent_id and old_parent_id in span_id_map:
                        new_span["parent_span_id"] = span_id_map[old_parent_id]
                    else:
                        new_span["parent_span_id"] = old_parent_id

                # Inject labels directly into attributes, preserving existing ones
                if "attributes" not in new_span:
                    new_span["attributes"] = {}

                for k, v in labels.items():
                    new_span["attributes"][k] = v

                new_spans.append(new_span)

            # Construct output file based on original baseline format
            output_content: Any
            if is_array_format:
                output_content = new_spans
            else:
                output_dict = copy.deepcopy(baseline)
                assert isinstance(output_dict, dict)
                output_dict["trace_id"] = new_trace_id
                if "span_count" in output_dict:
                    output_dict["span_count"] = len(new_spans)
                output_dict["spans"] = new_spans
                output_content = output_dict

            output_file = Path(args.output_dir) / f"trace_session_{session_idx}.json"
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_content, f, indent=2)
            except Exception as e:
                print(f"Error: Failed to write generated trace to '{output_file}': {e}", file=sys.stderr)
                return 1

            session_idx += 1

    print(f"Successfully generated {session_idx} sessions in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
