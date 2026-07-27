from evidencerank.agents.hallucination_checker import check_evidence
from evidencerank.models import EvidenceClaim, JudgeResult, Tier

RAW_CV_TEXT = "Daniel Taylor\nSkills: Python, Machine Learning\n5 years of Python experience"


def test_check_evidence_verifies_real_quote():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )

    report = check_evidence(judge_result, RAW_CV_TEXT)

    assert report.candidate_id == "c1"
    assert report.all_verified is True


def test_check_evidence_flags_fabricated_quote():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[
            EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience"),
            EvidenceClaim(claim="Led a team of 10 engineers", quote="managed a team of 10 engineers"),
        ],
    )

    report = check_evidence(judge_result, RAW_CV_TEXT)

    assert report.all_verified is False
    assert "managed a team of 10 engineers" in report.unverified_quotes
    assert "5 years of Python experience" not in report.unverified_quotes


def test_check_evidence_verifies_quote_despite_whitespace_differences():
    raw_cv_text = "Reduced latency by 40%"
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[
            EvidenceClaim(
                claim="Reduced latency",
                quote="Reduced\n\n\nlatency\n\n\nby\n\n\n40%",
            )
        ],
    )

    report = check_evidence(judge_result, raw_cv_text)

    assert report.all_verified is True
