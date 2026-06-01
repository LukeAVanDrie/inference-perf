import os

os.environ["COLUMNS"] = "150"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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

import pytest
from unittest.mock import Mock

from inference_perf.reportgen.base import ReportGenerator
from inference_perf.config import Config, ReportConfig, RequestLifecycleMetricsReportConfig, SessionLifecycleReportConfig
from inference_perf.apis.base import RequestLifecycleMetric, InferenceInfo, StreamedResponseMetrics
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.utils.report_file import ReportFile
from inference_perf.utils.cli_summary import print_sliced_summary_table


# Helper to mock a metric with labels
def _mock_metric_with_labels(labels: dict[str, str], stage_id: int = 0) -> RequestLifecycleMetric:
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(output_tokens=10, output_token_times=[1.0, 2.0, 3.0]),
        labels=labels,
    )
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=10.0,
        request_data="test_request",
        info=info,
        error=None,
    )


@pytest.mark.asyncio
async def test_cartesian_metric_slicing() -> None:
    # 1. Create mocked metrics with different labels
    m1 = _mock_metric_with_labels({"priority": "premium", "tenant_id": "tenant-a"})
    m2 = _mock_metric_with_labels({"priority": "premium", "tenant_id": "tenant-a"})
    m3 = _mock_metric_with_labels({"priority": "standard", "tenant_id": "tenant-b"})
    m4 = _mock_metric_with_labels({"priority": "standard"})  # missing tenant_id
    m5 = _mock_metric_with_labels({"tenant_id": "tenant-c"})  # missing priority

    metrics = [m1, m2, m3, m4, m5]

    # 2. Setup mock collector
    mock_collector = Mock(spec=RequestMetricCollector)
    mock_collector.get_metrics.return_value = metrics

    # 3. Setup Config with group_by_labels
    config = Config()
    report_config = ReportConfig(
        request_lifecycle=RequestLifecycleMetricsReportConfig(
            summary=False,  # disable main summary to focus on slices
            per_stage=False,
            per_request=False,
            per_adapter=False,
            group_by_labels=[["priority", "tenant_id"]],
        ),
        prometheus=None,
        session_lifecycle=SessionLifecycleReportConfig(summary=False, per_stage=False, per_session=False),
    )
    config.report = report_config

    # 4. Instantiate ReportGenerator
    reportgen = ReportGenerator(
        metrics_client=None,
        metrics_collector=mock_collector,
        config=config,
    )

    # 5. Run generate_reports
    runtime_params = Mock()
    reports = await reportgen.generate_reports(report_config, runtime_params)

    # 6. Assert reports are generated correctly
    # Expected slices:
    # ("premium", "tenant-a") -> m1, m2
    # ("standard", "tenant-b") -> m3
    # ("standard", "default") -> m4
    # ("default", "tenant-c") -> m5

    expected_names = {
        "summary_labels_premium_tenant-a_lifecycle_metrics",
        "summary_labels_standard_tenant-b_lifecycle_metrics",
        "summary_labels_standard_default_lifecycle_metrics",
        "summary_labels_default_tenant-c_lifecycle_metrics",
        "config",  # always generated
    }

    generated_names = {r.name for r in reports}
    assert generated_names == expected_names

    # Verify contents of one of the slices (e.g. premium_tenant-a)
    premium_report = next(r for r in reports if r.name == "summary_labels_premium_tenant-a_lifecycle_metrics")
    assert premium_report.contents["labels"] == {"priority": "premium", "tenant_id": "tenant-a"}
    assert premium_report.contents["load_summary"]["count"] == 2

    # Verify fallback default slice
    standard_default_report = next(r for r in reports if r.name == "summary_labels_standard_default_lifecycle_metrics")
    assert standard_default_report.contents["labels"] == {"priority": "standard", "tenant_id": "default"}
    assert standard_default_report.contents["load_summary"]["count"] == 1


def test_cli_summary_table_empty() -> None:
    # If no sliced reports, should return silently (no error, no table printed)
    reports = [ReportFile(name="config", contents={})]
    print_sliced_summary_table(reports)


def test_cli_summary_table_printing(capsys: pytest.CaptureFixture[str]) -> None:
    # Create some dummy sliced reports
    def _create_dummy_slice_report(labels: dict[str, str], count: int, qps: float, failed_count: int = 0) -> ReportFile:
        val_str = "_".join(labels.values())
        success_count = count - failed_count
        contents = {
            "labels": labels,
            "load_summary": {"count": count},
            "successes": {
                "count": success_count,
                "throughput": {"requests_per_sec": qps},
                "latency": {"time_to_first_token": {"mean": 0.045, "p90": 0.090}},
                "goodput_metrics": {"goodput_percentage": 98.5},
            },
            "failures": {"count": failed_count},
        }
        return ReportFile(name=f"summary_labels_{val_str}_lifecycle_metrics", contents=contents)

    reports = [
        _create_dummy_slice_report({"priority": "premium", "tenant_id": "tenant-a"}, 100, 10.5, failed_count=0),
        _create_dummy_slice_report({"priority": "standard", "tenant_id": "tenant-b"}, 50, 5.2, failed_count=5),
    ]

    # Call printing
    print_sliced_summary_table(reports)
    # Capture stdout
    captured = capsys.readouterr()
    assert "Sliced Performance Summary" in captured.out
    assert "Error %" in captured.out

    assert "priority=premium" in captured.out
    assert "tenant_id=tenant-a" in captured.out
    assert "100" in captured.out
    assert "10.5" in captured.out
    assert "0.0%" in captured.out

    assert "priority=standard" in captured.out
    assert "tenant_id=tenant-b" in captured.out
    assert "50" in captured.out
    assert "5.2" in captured.out
    assert "10.0%" in captured.out

    assert "45.0" in captured.out
    assert "90.0" in captured.out
    assert "98.5%" in captured.out


def test_cli_summary_table_truncation(capsys: pytest.CaptureFixture[str]) -> None:
    # Create 20 dummy sliced reports (exceeding 15)
    def _create_dummy_slice_report(labels: dict[str, str], count: int) -> ReportFile:
        val_str = "_".join(labels.values())
        contents = {
            "labels": labels,
            "load_summary": {"count": count},
            "successes": {
                "count": count,
                "throughput": {"requests_per_sec": 1.0},
                "latency": {"time_to_first_token": {"mean": 0.05, "p90": 0.1}},
            },
            "failures": {"count": 0},
        }
        return ReportFile(name=f"summary_labels_{val_str}_lifecycle_metrics", contents=contents)

    reports = []
    for i in range(20):
        count = (20 - i) * 10
        reports.append(_create_dummy_slice_report({"tenant": f"t{i}"}, count))

    # Call printing
    print_sliced_summary_table(reports)
    
    # Capture stdout
    captured = capsys.readouterr()
    assert "Sliced Performance Summary" in captured.out
    # Should print only top 15 (t0 to t14)
    for i in range(15):
        assert f"tenant=t{i}" in captured.out

    # Should NOT print t15 to t19
    for i in range(15, 20):
        assert f"tenant=t{i}" not in captured.out

    # Should print the truncation message
    assert "... and 5 more slices (see output JSON files for full details)" in captured.out
