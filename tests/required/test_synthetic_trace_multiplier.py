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

"""Unit and integration tests for OTel Trace Annotator & Multiplier."""

import json
import re
import subprocess
import tempfile
from pathlib import Path
import pytest

# Import internal parser logic
from scripts.annotate_and_multiply_traces import (
    parse_and_normalize_config,
    distribute_sessions,
)


def test_config_parsing_independent() -> None:
    """Verifies that independent configuration mode parses, calculates Cartesian product, and normalizes weights."""
    config = {
        "mode": "independent",
        "labels": {
            "priority": {
                "premium": 3,  # Implied weight 3/10 = 0.3
                "standard": 5,  # Implied weight 5/10 = 0.5
                "best-effort": 2,  # Implied weight 2/10 = 0.2
            },
            "tenant_id": {
                "tenant-a": 4,  # Implied weight 4/10 = 0.4
                "tenant-b": 6,  # Implied weight 6/10 = 0.6
            },
        },
    }
    slices = parse_and_normalize_config(config)

    # Expected slices: 3 * 2 = 6 slices
    assert len(slices) == 6

    # Verify total weight sum is exactly 1.0
    total_weight = sum(s["weight"] for s in slices)
    assert pytest.approx(total_weight) == 1.0

    # Verify individual joint weights
    # P(premium, tenant-a) = 0.3 * 0.4 = 0.12
    premium_a_slice = next(
        s for s in slices if s["labels"]["priority"] == "premium" and s["labels"]["tenant_id"] == "tenant-a"
    )
    assert pytest.approx(premium_a_slice["weight"]) == 0.12

    # P(best-effort, tenant-b) = 0.2 * 0.6 = 0.12
    be_b_slice = next(s for s in slices if s["labels"]["priority"] == "best-effort" and s["labels"]["tenant_id"] == "tenant-b")
    assert pytest.approx(be_b_slice["weight"]) == 0.12


def test_config_parsing_joint_normalization() -> None:
    """Verifies that joint configuration mode normalizes slice weights correctly."""
    config = {
        "mode": "joint",
        "slices": [
            {"labels": {"priority": "premium", "tenant_id": "tenant-a"}, "weight": 1.5},
            {"labels": {"priority": "best-effort", "tenant_id": "tenant-b"}, "weight": 0.5},
        ],
    }
    slices = parse_and_normalize_config(config)

    assert len(slices) == 2
    # Weights normalized to: 1.5 / 2.0 = 0.75, and 0.5 / 2.0 = 0.25
    assert pytest.approx(slices[0]["weight"]) == 0.75
    assert pytest.approx(slices[1]["weight"]) == 0.25


def test_config_parsing_errors() -> None:
    """Verifies that parser throws ValueErrors for invalid configuration schemas."""
    # Zero weights for labels
    invalid_independent = {"mode": "independent", "labels": {"priority": {"premium": 0.0, "standard": 0.0}}}
    with pytest.raises(ValueError, match="Total weight for label 'priority' cannot be zero."):
        parse_and_normalize_config(invalid_independent)

    # Zero weights for joint slices
    invalid_joint = {
        "mode": "joint",
        "slices": [{"labels": {"priority": "premium"}, "weight": 0.0}, {"labels": {"priority": "best-effort"}, "weight": 0.0}],
    }
    with pytest.raises(ValueError, match="Total weight of distribution cannot be zero."):
        parse_and_normalize_config(invalid_joint)

    # Missing mode key
    with pytest.raises(ValueError, match="Config is missing required 'mode' key."):
        parse_and_normalize_config({})


def test_distribute_sessions() -> None:
    """Verifies that session count distribution using Largest Remainder Method sums exactly to num_sessions."""
    slices = [
        {"labels": {"priority": "premium"}, "weight": 0.35},
        {"labels": {"priority": "standard"}, "weight": 0.45},
        {"labels": {"priority": "best-effort"}, "weight": 0.20},
    ]

    # Total sessions = 100:
    # 35, 45, 20 (clean division)
    counts_100 = distribute_sessions(slices, 100)
    assert counts_100 == [35, 45, 20]
    assert sum(counts_100) == 100

    # Total sessions = 10:
    # 3.5 (allocated 3, remainder 0.5) -> becomes 4
    # 4.5 (allocated 4, remainder 0.5) -> becomes 4 or 5 depending on sort order index
    # 2.0 (allocated 2, remainder 0.0) -> remains 2
    # Total remainder distribution: missing = 10 - 9 = 1.
    # Index sorting by remainder: premium (0.5), standard (0.5), best-effort (0.0).
    # For equal remainders, Python's Timsort preserves stable sort order.
    # indices: 0 (premium), 1 (standard), 2 (best-effort)
    # missing = 1 allocated to index 0
    counts_10 = distribute_sessions(slices, 10)
    assert counts_10 == [4, 4, 2]
    assert sum(counts_10) == 10


def test_trace_generation_correctness() -> None:
    """Integration test executing the CLI script via subprocess, verifying output files count, DAG integrity, and label distribution."""
    # Build a self-contained, complex trace baseline in memory
    baseline_data = {
        "trace_id": "original_trace_parent_id",
        "collected_at": "2026-02-25T10:00:00.000000",
        "span_count": 3,
        "spans": [
            {
                "trace_id": "original_trace_parent_id",
                "span_id": "span_root",
                "parent_span_id": None,
                "name": "root_span",
                "attributes": {"custom_original_attr": "value_1"},
            },
            {
                "trace_id": "original_trace_parent_id",
                "span_id": "span_child_1",
                "parent_span_id": "span_root",
                "name": "child_span_1",
                "attributes": {},
            },
            {
                "trace_id": "original_trace_parent_id",
                "span_id": "span_child_2",
                "parent_span_id": "span_root",
                "name": "child_span_2",
                "attributes": {"nested_data": True},
            },
        ],
    }

    config_data = {
        "mode": "joint",
        "slices": [
            {"labels": {"priority": "premium", "tenant_id": "tenant-a"}, "weight": 0.6},
            {"labels": {"priority": "best-effort", "tenant_id": "tenant-b"}, "weight": 0.4},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_file = Path(tmpdir) / "baseline.json"
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f)

        config_file = Path(tmpdir) / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        output_dir = Path(tmpdir) / "output"
        num_sessions = 50

        # Run script via subprocess
        cmd = [
            "python3",
            "scripts/annotate_and_multiply_traces.py",
            "--baseline-file",
            str(baseline_file),
            "--output-dir",
            str(output_dir),
            "--num-sessions",
            str(num_sessions),
            "--labels-config",
            str(config_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"CLI execution failed. Stderr:\n{result.stderr}\nStdout:\n{result.stdout}"

        # Verify generated files count
        generated_files = list(output_dir.glob("trace_session_*.json"))
        assert len(generated_files) == num_sessions, f"Expected {num_sessions} session files, got {len(generated_files)}"

        premium_count = 0
        best_effort_count = 0

        for session_file in generated_files:
            with open(session_file, "r", encoding="utf-8") as f:
                trace = json.load(f)

            # 1. Verify trace-level fields are preserved and updated
            assert "trace_id" in trace
            assert trace["trace_id"] != "original_trace_parent_id"
            assert re.match(r"^[0-9a-f]{32}$", trace["trace_id"]), f"Trace ID {trace['trace_id']} is not 32 hex characters"
            assert trace["collected_at"] == "2026-02-25T10:00:00.000000"
            assert trace["span_count"] == 3

            spans = trace.get("spans", [])
            assert len(spans) == 3

            # 2. Verify span-level IDs, trace_ids, and parent relationship preservation
            span_root = next(s for s in spans if s["name"] == "root_span")
            span_child1 = next(s for s in spans if s["name"] == "child_span_1")
            span_child2 = next(s for s in spans if s["name"] == "child_span_2")

            # Verify IDs are unique and matches OTel specifications (16 hex chars)
            for span in [span_root, span_child1, span_child2]:
                assert span["trace_id"] == trace["trace_id"]
                assert re.match(r"^[0-9a-f]{16}$", span["span_id"])

            # Root span has no parent
            assert span_root.get("parent_span_id") is None

            # Child spans have their parents properly updated to the newly generated root span ID
            assert span_child1["parent_span_id"] == span_root["span_id"]
            assert span_child2["parent_span_id"] == span_root["span_id"]

            # Verify original custom attributes are preserved
            assert span_root["attributes"]["custom_original_attr"] == "value_1"
            assert span_child2["attributes"]["nested_data"] is True

            # 3. Check custom labels are correctly injected and consistent across session spans
            assert "priority" in span_root["attributes"]
            assert "tenant_id" in span_root["attributes"]

            priority = span_root["attributes"]["priority"]
            tenant_id = span_root["attributes"]["tenant_id"]

            for span in [span_root, span_child1, span_child2]:
                assert span["attributes"]["priority"] == priority
                assert span["attributes"]["tenant_id"] == tenant_id

            if priority == "premium":
                premium_count += 1
                assert tenant_id == "tenant-a"
            elif priority == "best-effort":
                best_effort_count += 1
                assert tenant_id == "tenant-b"
            else:
                pytest.fail(f"Unexpected priority: {priority}")

        # 4. Verify distribution ratios match weight configuration:
        # premium slices: 0.6 * 50 = 30 sessions
        # best-effort slices: 0.4 * 50 = 20 sessions
        assert premium_count == 30
        assert best_effort_count == 20


def test_num_sessions_validation() -> None:
    """Verifies that providing --num-sessions < 1 raises an error."""
    cmd = [
        "python3",
        "scripts/annotate_and_multiply_traces.py",
        "--baseline-file",
        "dummy.json",
        "--output-dir",
        "dummy_out",
        "--num-sessions",
        "0",
        "--labels-config",
        "dummy_config",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "must be at least 1" in result.stderr


def test_clean_flag_behavior() -> None:
    """Verifies that --clean clears the output directory, and default (False) preserves it."""
    baseline_data = [{"span_id": "1", "trace_id": "1", "name": "span1"}]
    config_data = {"mode": "joint", "slices": [{"labels": {"tenant": "a"}, "weight": 1.0}]}

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_file = Path(tmpdir) / "baseline.json"
        with open(baseline_file, "w") as f:
            json.dump(baseline_data, f)
        config_file = Path(tmpdir) / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Create some pre-existing files in output_dir
        file1 = output_dir / "pre_existing_1.txt"
        file1.write_text("hello")
        file2 = output_dir / "pre_existing_2.json"
        file2.write_text("{}")

        # 1. Run WITHOUT --clean (default is False)
        cmd = [
            "python3",
            "scripts/annotate_and_multiply_traces.py",
            "--baseline-file",
            str(baseline_file),
            "--output-dir",
            str(output_dir),
            "--num-sessions",
            "1",
            "--labels-config",
            str(config_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        # Verify pre-existing files STILL EXIST, plus the new trace file
        assert file1.exists()
        assert file2.exists()
        assert (output_dir / "trace_session_0.json").exists()

        # 2. Run WITH --clean
        cmd_clean = cmd + ["--clean"]
        result_clean = subprocess.run(cmd_clean, capture_output=True, text=True, check=False)
        assert result_clean.returncode == 0

        # Verify pre-existing files ARE GONE
        assert not file1.exists()
        assert not file2.exists()
        # But the new trace file exists
        assert (output_dir / "trace_session_0.json").exists()


def test_low_session_warning() -> None:
    """Verifies that a warning is logged to stderr when sessions are too low to represent all slices."""
    baseline_data = [{"span_id": "1", "trace_id": "1", "name": "span1"}]
    config_data = {
        "mode": "joint",
        "slices": [
            {"labels": {"priority": "premium", "tenant_id": "tenant-a"}, "weight": 0.4},
            {"labels": {"priority": "best-effort", "tenant_id": "tenant-b"}, "weight": 0.3},
            {"labels": {"priority": "standard", "tenant_id": "tenant-c"}, "weight": 0.3},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_file = Path(tmpdir) / "baseline.json"
        with open(baseline_file, "w") as f:
            json.dump(baseline_data, f)
        config_file = Path(tmpdir) / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        output_dir = Path(tmpdir) / "output"

        # Run with num_sessions = 2 (less than 3 slices)
        cmd = [
            "python3",
            "scripts/annotate_and_multiply_traces.py",
            "--baseline-file",
            str(baseline_file),
            "--output-dir",
            str(output_dir),
            "--num-sessions",
            "2",
            "--labels-config",
            str(config_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        # Verify warning in stderr
        assert "WARNING" in result.stderr
        assert "--num-sessions (2) is less than the number of configured slices (3)" in result.stderr
        assert "The following slices will receive 0 sessions:" in result.stderr

        # tenant-c (index 2) should receive 0 sessions.
        # Labels are sorted: "priority=standard, tenant_id=tenant-c"
        assert "- priority=standard, tenant_id=tenant-c" in result.stderr
        assert "- priority=premium" not in result.stderr
        assert "- priority=best-effort" not in result.stderr
