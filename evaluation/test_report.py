import json
from pathlib import Path
from unittest.mock import Mock

from evaluation.metrics import (
    evidence_relevancy_metric,
    groundedness_metric,
    recruiter_alignment_metric,
)
from evaluation.report import compute_geval_scores, compute_pipeline_stats


def _write_report(path: Path, **overrides) -> None:
    base = {
        "jd": {
            "title": "ML Engineer",
            "required_skills": ["Python", "PyTorch"],
            "nice_to_have_skills": ["Docker"],
            "min_experience_years": 2,
            "education": "",
            "responsibilities": ["Build models"],
        },
        "profiles": {},
        "prefilter_results": {},
        "dropped": [],
        "judge_results": {},
        "calibrated_results": [],
        "hallucination_reports": {},
    }
    base.update(overrides)
    path.write_text(json.dumps(base), encoding="utf-8")


def test_compute_pipeline_stats_counts_candidates(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={
            "alice": {"raw_cv_text": "alice cv"},
            "bob": {"raw_cv_text": "bob cv"},
            "carol": {"raw_cv_text": "carol cv"},
        },
        dropped=[{"candidate_id": "carol", "reason": "pre-filter: no relevant skill overlap"}],
        judge_results={
            "alice": {"tier": "Strong Fit", "rating": 9, "evidence": []},
            "bob": {"tier": "Weak Fit", "rating": 3, "evidence": []},
        },
        hallucination_reports={
            "alice": {"candidate_id": "alice", "unverified_quotes": []},
            "bob": {"candidate_id": "bob", "unverified_quotes": ["some quote"]},
        },
    )

    stats = compute_pipeline_stats(report_path)

    assert stats == {
        "total_candidates": 3,
        "passed_prefilter": 2,
        "dropped_prefilter": 1,
        "evaluated_by_judge": 2,
        "hallucination_flagged": 1,
    }


def _write_geval_report(path: Path, judge_results: dict, profiles: dict | None = None) -> None:
    _write_report(
        path,
        profiles=profiles
        or {candidate_id: {"raw_cv_text": f"{candidate_id} cv"} for candidate_id in judge_results},
        judge_results=judge_results,
    )


def test_compute_geval_scores_aggregates_two_candidates(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_geval_report(
        report_path,
        judge_results={
            "alice": {"tier": "Strong Fit", "rating": 9, "evidence": [{"claim": "c1", "quote": "q1"}]},
            "bob": {"tier": "Weak Fit", "rating": 3, "evidence": [{"claim": "c2", "quote": "q2"}]},
        },
    )

    monkeypatch.setattr(groundedness_metric, "measure", Mock(side_effect=[0.9, 0.5]))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(side_effect=[0.8, 0.4]))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(side_effect=[1.0, 0.6]))

    scores = compute_geval_scores(report_path)

    assert scores["Groundedness"]["n"] == 2
    assert round(scores["Groundedness"]["mean"], 4) == 0.7
    assert round(scores["Groundedness"]["std"], 4) == 0.2828  # stdev([0.9, 0.5])
    assert scores["Groundedness"]["pass_rate"] == 0.5  # only 0.9 >= 0.7 threshold

    assert round(scores["RecruiterAlignment"]["mean"], 4) == 0.6
    assert scores["EvidenceRelevancy"]["pass_rate"] == 0.5  # only 1.0 >= 0.7 threshold


def test_compute_geval_scores_empty_judge_results_returns_none_fields(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_geval_report(report_path, judge_results={})

    mock = Mock()
    monkeypatch.setattr(groundedness_metric, "measure", mock)
    monkeypatch.setattr(recruiter_alignment_metric, "measure", mock)
    monkeypatch.setattr(evidence_relevancy_metric, "measure", mock)

    scores = compute_geval_scores(report_path)

    assert scores["Groundedness"] == {"n": 0, "mean": None, "std": None, "pass_rate": None}
    mock.assert_not_called()


def test_compute_geval_scores_single_candidate_std_is_none(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_geval_report(
        report_path,
        judge_results={"alice": {"tier": "Weak Fit", "rating": 4, "evidence": []}},
    )

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.6))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.6))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.6))

    scores = compute_geval_scores(report_path)

    assert scores["Groundedness"]["n"] == 1
    assert scores["Groundedness"]["mean"] == 0.6
    assert scores["Groundedness"]["std"] is None
    assert scores["Groundedness"]["pass_rate"] == 0.0  # 0.6 < 0.7 threshold


def _write_calibrated_report(path: Path, ranks: dict[str, int]) -> None:
    _write_geval_report(
        path,
        judge_results={
            candidate_id: {"tier": "Strong Fit", "rating": 8, "evidence": []}
            for candidate_id in ranks
        },
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["calibrated_results"] = [
        {
            "candidate_id": candidate_id,
            "final_rank": final_rank,
            "tier": "Strong Fit",
            "rating": 8,
            "calibration_notes": "",
        }
        for candidate_id, final_rank in ranks.items()
    ]
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_eval_markdown_report_single_run_omits_rank_stability(tmp_path, monkeypatch):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1, "bob": 2})

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    markdown = build_eval_markdown_report([report_path])

    assert "## Pipeline Stats" in markdown
    assert "## GEval Metrics" in markdown
    assert "## Rank Stability" not in markdown


def test_build_eval_markdown_report_multi_run_includes_rank_stability(tmp_path, monkeypatch):
    from evaluation.report import build_eval_markdown_report

    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_calibrated_report(report_a, {"alice": 1, "bob": 2})
    _write_calibrated_report(report_b, {"alice": 1, "bob": 2})

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    markdown = build_eval_markdown_report([report_a, report_b])

    assert "## Rank Stability" in markdown
    assert "1.000" in markdown  # identical rankings -> spearman/kendall == 1.0


def test_write_eval_markdown_report_writes_file(tmp_path, monkeypatch):
    from evaluation.report import write_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    out_path = tmp_path / "eval_report.md"

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    write_eval_markdown_report([report_path], out_path)

    assert out_path.exists()
    assert "## Pipeline Stats" in out_path.read_text(encoding="utf-8")
