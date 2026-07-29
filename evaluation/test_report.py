import json
from pathlib import Path

import pytest

from evidencerank.models import (
    CalibratedResult,
    CandidateProfile,
    ContactInfo,
    EvidenceClaim,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    PrefilterResult,
    Tier,
)

from evaluation.report import (
    build_json_report,
    build_markdown_report,
    compute_pipeline_stats,
    load_rank_map,
    rank_stability,
    write_json_report,
    write_markdown_report,
)


def _sample_state():
    return {
        "jd": JDRequirements(title="ML Engineer", required_skills=["Python"]),
        "profiles": {
            "strong": CandidateProfile(
                candidate_id="strong",
                raw_cv_text="Jane Doe, 5 years Python",
                contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
                skills=["Python"],
            )
        },
        "prefilter_results": {
            "strong": PrefilterResult(candidate_id="strong", similarity=0.9, passed=True),
            "weak": PrefilterResult(candidate_id="weak", similarity=0.1, passed=False),
        },
        "dropped": [{"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}],
        "judge_results": {
            "strong": JudgeResult(
                candidate_id="strong", tier=Tier.STRONG_FIT, rating=9,
                evidence=[EvidenceClaim(claim="Strong Python background", quote="5 years Python")],
            )
        },
        "calibrated_results": [
            CalibratedResult(
                candidate_id="strong", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Only surviving candidate",
            )
        ],
        "hallucination_reports": {
            "strong": HallucinationReport(candidate_id="strong", unverified_quotes=[]),
        },
    }


def test_build_json_report_contains_all_sections():
    report = build_json_report(_sample_state())

    assert report["jd"]["title"] == "ML Engineer"
    assert report["profiles"]["strong"]["candidate_id"] == "strong"
    assert report["profiles"]["strong"]["raw_cv_text"] == "Jane Doe, 5 years Python"
    assert report["profiles"]["strong"]["contact"]["name"] == "Jane Doe"
    assert report["prefilter_results"]["strong"]["similarity"] == 0.9
    assert report["prefilter_results"]["strong"]["passed"] is True
    assert report["prefilter_results"]["weak"]["passed"] is False
    assert report["dropped"][0]["candidate_id"] == "weak"
    assert report["judge_results"]["strong"]["rating"] == 9
    assert report["calibrated_results"][0]["final_rank"] == 1
    assert report["hallucination_reports"]["strong"]["unverified_quotes"] == []


def test_build_json_report_defaults_missing_stages_to_empty():
    minimal_state = {"jd": JDRequirements(title="ML Engineer", required_skills=["Python"])}

    report = build_json_report(minimal_state)

    assert report["profiles"] == {}
    assert report["prefilter_results"] == {}


def test_build_json_report_includes_stage_timings():
    state = _sample_state()
    state["stage_timings"] = {"extract_profiles": 1.5, "judge": 3.25}

    report = build_json_report(state)

    assert report["stage_timings"] == {"extract_profiles": 1.5, "judge": 3.25}


def test_build_json_report_defaults_missing_stage_timings_to_empty_dict():
    report = build_json_report(_sample_state())

    assert report["stage_timings"] == {}


def test_write_json_report_writes_valid_json(tmp_path):
    out_path = tmp_path / "report.json"
    write_json_report(_sample_state(), out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["calibrated_results"][0]["candidate_id"] == "strong"


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
        "mean_evidence_relevancy": 0.0,
    }


def test_compute_pipeline_stats_hallucination_rate_is_zero_when_no_one_judged(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, profiles={}, judge_results={}, hallucination_reports={})

    stats = compute_pipeline_stats(report_path)

    assert stats["hallucination_rate"] == 0.0


class _FakeRelevancyEmbedder:
    def encode(self, texts):
        import numpy as np

        vectors = []
        for text in texts:
            if "Python" in text:
                vectors.append([1.0, 0.0])
            elif "Baking" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.array(vectors)


def test_compute_pipeline_stats_evidence_relevancy_scores_relevant_claims_higher(monkeypatch, tmp_path):
    monkeypatch.setattr("evaluation.report._get_embedder", lambda: _FakeRelevancyEmbedder())

    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        jd={
            "title": "ML Engineer", "required_skills": ["Python"], "nice_to_have_skills": [],
            "min_experience_years": 0, "education": "", "responsibilities": [],
        },
        profiles={"relevant": {"raw_cv_text": "x"}, "irrelevant": {"raw_cv_text": "y"}},
        judge_results={
            "relevant": {
                "tier": "Strong Fit", "rating": 9,
                "evidence": [{"claim": "Strong Python background", "quote": "5 years Python"}],
            },
            "irrelevant": {
                "tier": "Weak Fit", "rating": 2,
                "evidence": [{"claim": "Extensive Baking experience", "quote": "Baking pastries"}],
            },
        },
    )

    stats = compute_pipeline_stats(report_path)

    assert stats["mean_evidence_relevancy"] == 0.5


def test_compute_pipeline_stats_evidence_relevancy_zero_when_no_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr("evaluation.report._get_embedder", lambda: _FakeRelevancyEmbedder())

    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"alice": {"raw_cv_text": "x"}},
        judge_results={"alice": {"tier": "Strong Fit", "rating": 9, "evidence": []}},
    )

    stats = compute_pipeline_stats(report_path)

    assert stats["mean_evidence_relevancy"] == 0.0


def test_compute_pipeline_stats_evidence_relevancy_zero_when_jd_has_no_reference_text(monkeypatch, tmp_path):
    monkeypatch.setattr("evaluation.report._get_embedder", lambda: _FakeRelevancyEmbedder())

    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        jd={
            "title": "ML Engineer", "required_skills": [], "nice_to_have_skills": [],
            "min_experience_years": 0, "education": "", "responsibilities": [],
        },
        profiles={"alice": {"raw_cv_text": "x"}},
        judge_results={
            "alice": {
                "tier": "Strong Fit", "rating": 9,
                "evidence": [{"claim": "Strong Python background", "quote": "5 years Python"}],
            }
        },
    )

    stats = compute_pipeline_stats(report_path)

    assert stats["mean_evidence_relevancy"] == 0.0


def test_build_markdown_report_includes_evidence_relevancy_row(monkeypatch, tmp_path):
    monkeypatch.setattr("evaluation.report._get_embedder", lambda: _FakeRelevancyEmbedder())

    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        jd={
            "title": "ML Engineer", "required_skills": ["Python"], "nice_to_have_skills": [],
            "min_experience_years": 0, "education": "", "responsibilities": [],
        },
        profiles={"alice": {"raw_cv_text": "x"}},
        judge_results={
            "alice": {
                "tier": "Strong Fit", "rating": 9,
                "evidence": [{"claim": "Strong Python background", "quote": "5 years Python"}],
            }
        },
        calibrated_results=[
            {"candidate_id": "alice", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
    )

    markdown = build_markdown_report([report_path])

    assert "| Evidence Relevancy | 1.000 |" in markdown


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


def test_build_markdown_report_includes_rankings_and_pipeline_stats(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1, "bob": 2})

    markdown = build_markdown_report([report_path])

    assert "## Rankings" in markdown
    assert (
        "| Rank | Candidate | Tier | Rating | Key Evidence | Hallucination Flags "
        "| Calibration Notes |" in markdown
    )
    assert "| 1 | alice | Strong Fit | 8 |  | — |  |" in markdown
    assert "## Pipeline Stats" in markdown


def test_build_markdown_report_pipeline_stats_is_mean_across_multiple_reports(tmp_path):
    report_a = tmp_path / "report_a.json"
    _write_report(
        report_a,
        profiles={c: {"raw_cv_text": c} for c in ["a1", "a2", "a3", "a4"]},
        dropped=[
            {"candidate_id": "a3", "reason": "x"},
            {"candidate_id": "a4", "reason": "x"},
        ],
        judge_results={
            "a1": {"tier": "Strong Fit", "rating": 9, "evidence": []},
            "a2": {"tier": "Strong Fit", "rating": 8, "evidence": []},
        },
        hallucination_reports={
            "a1": {"candidate_id": "a1", "unverified_quotes": []},
            "a2": {"candidate_id": "a2", "unverified_quotes": []},
        },
        calibrated_results=[
            {"candidate_id": "a1", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""},
            {"candidate_id": "a2", "final_rank": 2, "tier": "Strong Fit", "rating": 8, "calibration_notes": ""},
        ],
    )

    report_b = tmp_path / "report_b.json"
    _write_report(
        report_b,
        profiles={c: {"raw_cv_text": c} for c in ["a1", "a2", "a3", "a4"]},
        dropped=[],
        judge_results={
            c: {"tier": "Strong Fit", "rating": 8, "evidence": []} for c in ["a1", "a2", "a3", "a4"]
        },
        hallucination_reports={
            "a1": {"candidate_id": "a1", "unverified_quotes": ["fabricated"]},
            "a2": {"candidate_id": "a2", "unverified_quotes": ["fabricated"]},
            "a3": {"candidate_id": "a3", "unverified_quotes": []},
            "a4": {"candidate_id": "a4", "unverified_quotes": []},
        },
        calibrated_results=[
            {"candidate_id": "a1", "final_rank": 1, "tier": "Strong Fit", "rating": 8, "calibration_notes": ""},
            {"candidate_id": "a2", "final_rank": 2, "tier": "Strong Fit", "rating": 8, "calibration_notes": ""},
        ],
    )

    markdown = build_markdown_report([report_a, report_b])

    assert f"| {report_a} | 4 | 2 | 2 | 2 | 0.0% |" in markdown
    assert f"| {report_b} | 4 | 4 | 0 | 4 | 50.0% |" in markdown
    assert "| **Mean** | 4 | 3 | 1 | 3 | 25.0% |" in markdown


def test_build_markdown_report_single_run_omits_rank_stability(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1, "bob": 2})

    markdown = build_markdown_report([report_path])

    assert "## Rank Stability" not in markdown


def test_build_markdown_report_multi_run_includes_rank_stability(tmp_path):
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_calibrated_report(report_a, {"alice": 1, "bob": 2})
    _write_calibrated_report(report_b, {"alice": 1, "bob": 2})

    markdown = build_markdown_report([report_a, report_b])

    assert "## Rank Stability" in markdown
    assert "1.000" in markdown


def test_write_markdown_report_writes_file(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    out_path = tmp_path / "report.md"

    write_markdown_report([report_path], out_path)

    assert out_path.exists()
    assert "## Pipeline Stats" in out_path.read_text(encoding="utf-8")


def test_build_markdown_report_includes_stage_timings_when_present(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["stage_timings"] = {"extract_profiles": 1.5, "judge": 3.25}
    report_path.write_text(json.dumps(data), encoding="utf-8")

    markdown = build_markdown_report([report_path])

    assert "## Stage Timings" in markdown
    assert "| extract_profiles | 1.500 |" in markdown
    assert "| judge | 3.250 |" in markdown


def test_build_markdown_report_omits_stage_timings_when_absent(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})

    markdown = build_markdown_report([report_path])

    assert "## Stage Timings" not in markdown


def test_build_markdown_report_orders_rankings_by_rank_ascending(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={
            "first": {"raw_cv_text": "first cv"},
            "second": {"raw_cv_text": "second cv"},
            "third": {"raw_cv_text": "third cv"},
        },
        calibrated_results=[
            {
                "candidate_id": "third", "final_rank": 3, "tier": "Weak Fit",
                "rating": 4, "calibration_notes": "Ranked third",
            },
            {
                "candidate_id": "first", "final_rank": 1, "tier": "Strong Fit",
                "rating": 9, "calibration_notes": "Ranked first",
            },
            {
                "candidate_id": "second", "final_rank": 2, "tier": "Moderate Fit",
                "rating": 6, "calibration_notes": "Ranked second",
            },
        ],
    )

    markdown = build_markdown_report([report_path])
    lines = markdown.splitlines()

    assert lines.index("| 1 | first | Strong Fit | 9 |  | — | Ranked first |") < \
        lines.index("| 2 | second | Moderate Fit | 6 |  | — | Ranked second |") < \
        lines.index("| 3 | third | Weak Fit | 4 |  | — | Ranked third |")


def test_build_markdown_report_escapes_pipes_and_newlines_in_notes(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        calibrated_results=[
            {
                "candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9,
                "calibration_notes": "Great fit | but watch out\nfor gaps in employment",
            }
        ],
    )

    markdown = build_markdown_report([report_path])
    data_row = next(line for line in markdown.splitlines() if line.startswith("| 1 |"))

    assert "\n" not in data_row
    assert "Great fit \\| but watch out for gaps in employment" in data_row
    assert "Great fit | but" not in data_row

    unescaped_split = data_row.replace("\\|", "").split("|")
    assert len(unescaped_split) == 9


def test_build_markdown_report_escapes_pipes_and_newlines_in_evidence(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        judge_results={
            "strong": {
                "tier": "Strong Fit", "rating": 9,
                "evidence": [
                    {"claim": "Led team", "quote": "Managed 5 | 10 person teams\nacross two years"},
                    {"claim": "Shipped feature", "quote": "Delivered on time"},
                ],
            }
        },
        calibrated_results=[
            {"candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
    )

    markdown = build_markdown_report([report_path])
    data_row = next(line for line in markdown.splitlines() if line.startswith("| 1 |"))

    assert "\n" not in data_row
    assert "Led team: Managed 5 \\| 10 person teams across two years" in data_row
    assert "Shipped feature: Delivered on time" in data_row
    assert "Managed 5 | 10 person" not in data_row

    unescaped_split = data_row.replace("\\|", "").split("|")
    assert len(unescaped_split) == 9


def test_build_markdown_report_shows_removed_count_for_flagged_candidate(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        calibrated_results=[
            {"candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
        hallucination_reports={
            "strong": {
                "candidate_id": "strong",
                "unverified_quotes": ["fabricated quote one", "fabricated quote two"],
            },
        },
    )

    markdown = build_markdown_report([report_path])

    assert "| 2 removed |" in markdown


def test_build_markdown_report_shows_dash_when_hallucination_report_has_no_unverified_quotes(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        calibrated_results=[
            {"candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
        hallucination_reports={
            "strong": {"candidate_id": "strong", "unverified_quotes": []},
        },
    )

    markdown = build_markdown_report([report_path])

    assert "| — |" in markdown


def test_build_markdown_report_tables_have_header_separator_rows(tmp_path):
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_calibrated_report(report_a, {"alice": 1, "bob": 2})
    _write_calibrated_report(report_b, {"alice": 1, "bob": 2})
    data = json.loads(report_a.read_text(encoding="utf-8"))
    data["stage_timings"] = {"extract_profiles": 1.5}
    report_a.write_text(json.dumps(data), encoding="utf-8")

    markdown = build_markdown_report([report_a, report_b])

    assert "| Rank | Candidate | Tier | Rating | Key Evidence | Hallucination Flags | Calibration Notes |\n|---|---|---|---|---|---|---|" in markdown
    assert (
        "| Run | Total candidates | Passed pre-filter | Dropped by pre-filter "
        "| Evaluated by Judge | Hallucination Rate | Evidence Relevancy |\n|---|---|---|---|---|---|---|"
    ) in markdown
    assert "| Stage | Seconds |\n|---|---|" in markdown
    assert "| Runs | Mean Spearman | Mean Kendall Tau |\n|---|---|---|" in markdown


def test_build_markdown_report_pipeline_stats_table_has_separator_row_for_single_run(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})

    markdown = build_markdown_report([report_path])

    assert "| Metric | Value |\n|---|---|" in markdown


def test_build_markdown_report_shows_dash_when_no_hallucination_report_present(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        calibrated_results=[
            {"candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
        hallucination_reports={},
    )

    markdown = build_markdown_report([report_path])

    assert "| — |" in markdown


def _write_rank_stability_report(path: Path, ranks: dict[str, int]) -> None:
    path.write_text(
        json.dumps({
            "calibrated_results": [
                {"candidate_id": candidate_id, "final_rank": final_rank}
                for candidate_id, final_rank in ranks.items()
            ]
        }),
        encoding="utf-8",
    )


def test_load_rank_map_reads_candidate_ranks(tmp_path):
    report_path = tmp_path / "run1.json"
    _write_rank_stability_report(report_path, {"a": 1, "b": 2})

    rank_map = load_rank_map(report_path)

    assert rank_map == {"a": 1, "b": 2}


def test_rank_stability_identical_rankings_scores_one(tmp_path):
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_rank_stability_report(run1, {"a": 1, "b": 2, "c": 3})
    _write_rank_stability_report(run2, {"a": 1, "b": 2, "c": 3})

    result = rank_stability([str(run1), str(run2)])

    assert result["mean_spearman"] == 1.0
    assert result["mean_kendall_tau"] == 1.0
    assert result["n_runs"] == 2


def test_rank_stability_reversed_rankings_scores_negative_one(tmp_path):
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_rank_stability_report(run1, {"a": 1, "b": 2, "c": 3})
    _write_rank_stability_report(run2, {"a": 3, "b": 2, "c": 1})

    result = rank_stability([str(run1), str(run2)])

    assert result["mean_spearman"] == -1.0


def test_rank_stability_intersects_candidate_ids_across_runs(tmp_path):
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_rank_stability_report(run1, {"a": 1, "b": 2, "c": 3, "d": 4})
    _write_rank_stability_report(run2, {"a": 1, "b": 2, "c": 3})

    result = rank_stability([str(run1), str(run2)])

    assert result["mean_spearman"] == 1.0
    assert result["mean_kendall_tau"] == 1.0
    assert result["n_runs"] == 2


def test_rank_stability_raises_when_fewer_than_two_candidates_are_common(tmp_path):
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_rank_stability_report(run1, {"a": 1, "b": 2})
    _write_rank_stability_report(run2, {"a": 1})

    with pytest.raises(ValueError):
        rank_stability([str(run1), str(run2)])
