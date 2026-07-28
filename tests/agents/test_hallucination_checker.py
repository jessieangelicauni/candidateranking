from evidencerank.agents.hallucination_checker import check_evidence, filter_verified_evidence
from evidencerank.models import (
    CandidateProfile,
    ContactInfo,
    EvidenceClaim,
    JudgeResult,
    Tier,
    WorkHistoryEntry,
)


def _profile(**overrides) -> CandidateProfile:
    defaults = dict(
        candidate_id="c1",
        raw_cv_text="Daniel Taylor\nSkills: Python, Machine Learning\n5 years of Python experience",
        contact=ContactInfo(name="Daniel Taylor"),
        skills=["Python", "Machine Learning"],
        work_history=[
            WorkHistoryEntry(
                title="Engineer", company="Acme Corp", start_date="2019", end_date="2022",
                achievements=["5 years of Python experience"],
            )
        ],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def test_check_evidence_verifies_quote_matching_extracted_achievement():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )

    report = check_evidence(judge_result, _profile())

    assert report.candidate_id == "c1"
    assert report.all_verified is True


def test_check_evidence_flags_quote_not_in_extracted_fields():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[
            EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience"),
            EvidenceClaim(claim="Led a team of 10 engineers", quote="managed a team of 10 engineers"),
        ],
    )

    report = check_evidence(judge_result, _profile())

    assert report.all_verified is False
    assert "managed a team of 10 engineers" in report.unverified_quotes
    assert "5 years of Python experience" not in report.unverified_quotes


def test_check_evidence_verifies_quote_despite_whitespace_differences():
    profile = _profile(
        work_history=[
            WorkHistoryEntry(
                title="Engineer", company="Acme Corp", start_date="2019", end_date="2022",
                achievements=["Reduced latency by 40%"],
            )
        ],
    )
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

    report = check_evidence(judge_result, profile)

    assert report.all_verified is True


def test_check_evidence_verifies_quote_matching_bare_skill():
    # A quote that's just one of the extracted skills tokens should verify too,
    # since skills are part of the extracted-fields text being checked against.
    profile = _profile(skills=["Python", "Kubernetes"])
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Kubernetes experience", quote="Kubernetes")],
    )

    report = check_evidence(judge_result, profile)

    assert report.all_verified is True


def test_filter_verified_evidence_removes_only_unverified_claims():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[
            EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience"),
            EvidenceClaim(claim="Led a team", quote="managed a team of 10 engineers"),
        ],
    )
    report = check_evidence(judge_result, _profile())

    filtered = filter_verified_evidence(judge_result, report)

    assert [claim.quote for claim in filtered.evidence] == ["5 years of Python experience"]
    assert filtered.candidate_id == "c1"
    assert filtered.tier == Tier.STRONG_FIT
    assert filtered.rating == 8


def test_filter_verified_evidence_returns_empty_list_when_all_unverified():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[
            EvidenceClaim(claim="Fabricated one", quote="totally fabricated quote one"),
            EvidenceClaim(claim="Fabricated two", quote="totally fabricated quote two"),
        ],
    )
    report = check_evidence(judge_result, _profile())

    filtered = filter_verified_evidence(judge_result, report)

    assert filtered.evidence == []
    assert filtered.candidate_id == "c1"
    assert filtered.tier == Tier.STRONG_FIT


def test_filter_verified_evidence_keeps_all_claims_when_fully_verified():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    report = check_evidence(judge_result, _profile())

    filtered = filter_verified_evidence(judge_result, report)

    assert filtered.evidence == judge_result.evidence
