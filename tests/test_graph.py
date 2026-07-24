from evidencerank.graph import build_graph
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


def test_graph_runs_extract_prefilter_judge_calibrate_hallucination(monkeypatch):
    jd = JDRequirements(title="ML Engineer", required_skills=["Python", "PyTorch"])

    # NOTE: "weak" is intentionally NOT last here. If hallucination_check_node
    # were to pair judge_results with profiles positionally (e.g. via zip)
    # instead of doing a proper dict lookup by candidate_id, the dropped
    # "weak" profile sitting in the middle would shift the positional
    # alignment and cause raw_cv_text to be mismatched between strong_a and
    # strong_b. Putting "weak" last would let such a bug pass by coincidence.
    raw_resumes = {
        "strong_a": "Python resume text - candidate A unique marker AAA",
        "weak": "Photoshop resume text",
        "strong_b": "Python resume text - candidate B unique marker BBB",
    }

    def fake_extract_cv(candidate_id, raw_text):
        return CandidateProfile(
            candidate_id=candidate_id,
            raw_cv_text=raw_text,
            contact=ContactInfo(name=candidate_id),
            skills=["Python"] if candidate_id != "weak" else ["Photoshop"],
        )

    def fake_prefilter_candidate(candidate_id, jd_required_skills, candidate_skills, threshold):
        passed = candidate_id != "weak"
        return PrefilterResult(candidate_id=candidate_id, similarity=0.9 if passed else 0.1, passed=passed)

    def fake_judge_candidate(jd_requirements, profile):
        return JudgeResult(
            candidate_id=profile.candidate_id,
            tier=Tier.STRONG_FIT,
            rating=9,
            evidence=[EvidenceClaim(claim="Strong fit", quote="Python")],
        )

    # Records every call to calibrate_pool so we can assert it is invoked
    # exactly once over the full surviving pool, not once per candidate.
    calibrate_calls: list[tuple[JDRequirements, list[JudgeResult]]] = []

    def fake_calibrate_pool(jd_requirements, judge_results):
        calibrate_calls.append((jd_requirements, list(judge_results)))
        return [
            CalibratedResult(
                candidate_id=r.candidate_id, final_rank=i + 1, tier=r.tier,
                rating=r.rating, calibration_notes="Ranked within pool",
            )
            for i, r in enumerate(judge_results)
        ]

    # Records the (candidate_id, raw_cv_text) pairs check_evidence was called
    # with, so we can assert each candidate was checked against its OWN raw
    # CV text rather than a mismatched/swapped one.
    hallucination_calls: list[tuple[str, str]] = []

    def fake_check_evidence(judge_result, raw_cv_text, threshold):
        hallucination_calls.append((judge_result.candidate_id, raw_cv_text))
        return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=[])

    monkeypatch.setattr("evidencerank.graph.extract_cv", fake_extract_cv)
    monkeypatch.setattr("evidencerank.graph.prefilter_candidate", fake_prefilter_candidate)
    monkeypatch.setattr("evidencerank.graph.judge_candidate", fake_judge_candidate)
    monkeypatch.setattr("evidencerank.graph.calibrate_pool", fake_calibrate_pool)
    monkeypatch.setattr("evidencerank.graph.check_evidence", fake_check_evidence)

    graph = build_graph()
    final_state = graph.invoke(
        {
            "jd": jd,
            "raw_resumes": raw_resumes,
        }
    )

    assert set(final_state["profiles"].keys()) == {"strong_a", "strong_b", "weak"}
    assert final_state["dropped"] == [
        {"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}
    ]
    assert set(final_state["judge_results"].keys()) == {"strong_a", "strong_b"}
    assert len(final_state["calibrated_results"]) == 2
    assert {r.candidate_id for r in final_state["calibrated_results"]} == {"strong_a", "strong_b"}

    # Regression guard 1: calibrate_pool must be invoked exactly once with
    # the full surviving pool, not once per candidate.
    assert len(calibrate_calls) == 1
    _, pooled_judge_results = calibrate_calls[0]
    assert {r.candidate_id for r in pooled_judge_results} == {"strong_a", "strong_b"}

    # Regression guard 2: each candidate's hallucination check must be run
    # against its OWN raw CV text, not a swapped/mismatched one.
    assert final_state["hallucination_reports"]["strong_a"].all_verified is True
    assert final_state["hallucination_reports"]["strong_b"].all_verified is True
    recorded_raw_text_by_candidate = dict(hallucination_calls)
    assert len(hallucination_calls) == 2
    for candidate_id in ("strong_a", "strong_b"):
        assert recorded_raw_text_by_candidate[candidate_id] == raw_resumes[candidate_id]
