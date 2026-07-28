import json
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner

from evaluation.cli import eval_report
from evaluation.metrics import (
    evidence_relevancy_metric,
    groundedness_metric,
    recruiter_alignment_metric,
)


def _write_minimal_report(path: Path) -> None:
    data = {
        "jd": {
            "title": "ML Engineer",
            "required_skills": ["Python"],
            "nice_to_have_skills": [],
            "min_experience_years": 0,
            "education": "",
            "responsibilities": [],
        },
        "profiles": {"alice": {"raw_cv_text": "alice cv"}},
        "prefilter_results": {},
        "dropped": [],
        "judge_results": {"alice": {"tier": "Strong Fit", "rating": 9, "evidence": []}},
        "calibrated_results": [
            {"candidate_id": "alice", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
        "hallucination_reports": {"alice": {"candidate_id": "alice", "unverified_quotes": []}},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_eval_report_cli_writes_output_file(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_minimal_report(report_path)
    out_path = tmp_path / "evaluation-metric.md"

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    runner = CliRunner()
    result = runner.invoke(
        eval_report, ["--reports", str(report_path), "--out", str(out_path)]
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert str(out_path) in result.output
