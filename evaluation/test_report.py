import json
from pathlib import Path

from evaluation.report import compute_pipeline_stats


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
        "hallucination_rate": 0.5,
    }


def test_compute_pipeline_stats_hallucination_rate_is_zero_when_no_one_judged(tmp_path):
    # Guards against a ZeroDivisionError when every candidate is dropped at
    # pre-filter and no one reaches the Judge.
    report_path = tmp_path / "report.json"
    _write_report(report_path, profiles={}, judge_results={}, hallucination_reports={})

    stats = compute_pipeline_stats(report_path)

    assert stats["hallucination_rate"] == 0.0


def _write_calibrated_report(path: Path, ranks: dict[str, int]) -> None:
    _write_report(
        path,
        profiles={candidate_id: {"raw_cv_text": f"{candidate_id} cv"} for candidate_id in ranks},
        judge_results={
            candidate_id: {"tier": "Strong Fit", "rating": 8, "evidence": []}
            for candidate_id in ranks
        },
        calibrated_results=[
            {
                "candidate_id": candidate_id,
                "final_rank": final_rank,
                "tier": "Strong Fit",
                "rating": 8,
                "calibration_notes": "",
            }
            for candidate_id, final_rank in ranks.items()
        ],
    )


def test_build_eval_markdown_report_single_run_omits_rank_stability(tmp_path):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1, "bob": 2})

    markdown = build_eval_markdown_report([report_path])

    assert "## Pipeline Stats" in markdown
    assert "## GEval Metrics" not in markdown
    assert "## Rank Stability" not in markdown


def test_build_eval_markdown_report_multi_run_includes_rank_stability(tmp_path):
    from evaluation.report import build_eval_markdown_report

    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_calibrated_report(report_a, {"alice": 1, "bob": 2})
    _write_calibrated_report(report_b, {"alice": 1, "bob": 2})

    markdown = build_eval_markdown_report([report_a, report_b])

    assert "## Rank Stability" in markdown
    assert "1.000" in markdown  # identical rankings -> spearman/kendall == 1.0


def test_write_eval_markdown_report_writes_file(tmp_path):
    from evaluation.report import write_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    out_path = tmp_path / "evaluation-metric.md"

    write_eval_markdown_report([report_path], out_path)

    assert out_path.exists()
    assert "## Pipeline Stats" in out_path.read_text(encoding="utf-8")


def test_build_eval_markdown_report_includes_stage_timings_when_present(tmp_path):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["stage_timings"] = {"extract_profiles": 1.5, "judge": 3.25}
    report_path.write_text(json.dumps(data), encoding="utf-8")

    markdown = build_eval_markdown_report([report_path])

    assert "## Stage Timings" in markdown
    assert "| extract_profiles | 1.500 |" in markdown
    assert "| judge | 3.250 |" in markdown


def test_build_eval_markdown_report_omits_stage_timings_when_absent(tmp_path):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    # NOTE: intentionally does NOT add "stage_timings" — simulates an older
    # report.json written before this key existed, to guard against a
    # KeyError regression.

    markdown = build_eval_markdown_report([report_path])

    assert "## Stage Timings" not in markdown
