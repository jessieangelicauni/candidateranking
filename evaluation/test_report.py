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
