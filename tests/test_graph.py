from unittest.mock import MagicMock

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

    def fake_extract_cv(candidate_id, raw_text):
        return CandidateProfile(
            candidate_id=candidate_id,
            raw_cv_text=raw_text,
            contact=ContactInfo(name=candidate_id),
            skills=["Python"] if candidate_id == "strong" else ["Photoshop"],
        )

    def fake_prefilter_candidate(candidate_id, jd_required_skills, candidate_skills, threshold):
        passed = candidate_id == "strong"
        return PrefilterResult(candidate_id=candidate_id, similarity=0.9 if passed else 0.1, passed=passed)

    def fake_judge_candidate(jd_requirements, profile):
        return JudgeResult(
            candidate_id=profile.candidate_id,
            tier=Tier.STRONG_FIT,
            rating=9,
            evidence=[EvidenceClaim(claim="Strong fit", quote="Python")],
        )

    def fake_calibrate_pool(jd_requirements, judge_results):
        return [
            CalibratedResult(
                candidate_id=r.candidate_id, final_rank=1, tier=r.tier,
                rating=r.rating, calibration_notes="Only candidate in pool",
            )
            for r in judge_results
        ]

    def fake_check_evidence(judge_result, raw_cv_text, threshold):
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
            "raw_resumes": {"strong": "Python resume text", "weak": "Photoshop resume text"},
        }
    )

    assert set(final_state["profiles"].keys()) == {"strong", "weak"}
    assert final_state["dropped"] == [
        {"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}
    ]
    assert set(final_state["judge_results"].keys()) == {"strong"}
    assert len(final_state["calibrated_results"]) == 1
    assert final_state["calibrated_results"][0].candidate_id == "strong"
    assert final_state["hallucination_reports"]["strong"].all_verified is True
