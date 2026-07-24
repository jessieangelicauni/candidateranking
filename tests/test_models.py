import pytest
from pydantic import ValidationError

from evidencerank.models import (
    CalibratedResult,
    CalibrationOutput,
    CandidateProfile,
    ContactInfo,
    EducationEntry,
    EvidenceClaim,
    ExtractedProfileFields,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    JudgeVerdict,
    PrefilterResult,
    ProjectEntry,
    Tier,
    WorkHistoryEntry,
)


def test_jd_requirements_defaults():
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    assert jd.nice_to_have_skills == []
    assert jd.min_experience_years == 0
    assert jd.responsibilities == []


def test_candidate_profile_inherits_extracted_fields():
    profile = CandidateProfile(
        candidate_id="c1",
        raw_cv_text="raw text",
        contact=ContactInfo(name="Jane Doe"),
        skills=["Python", "SQL"],
        work_history=[
            WorkHistoryEntry(
                title="ML Engineer", company="Acme", start_date="2020",
                end_date="2023", achievements=["Shipped model"],
            )
        ],
        education=[EducationEntry(degree="BSc", institution="MIT", year="2019")],
        projects=[ProjectEntry(name="Recommender", description="...", tech=["Python"])],
    )
    assert isinstance(profile, ExtractedProfileFields)
    assert profile.candidate_id == "c1"
    assert profile.contact.name == "Jane Doe"


def test_judge_result_rating_bounds():
    with pytest.raises(ValidationError):
        JudgeVerdict(tier=Tier.STRONG_FIT, rating=11, evidence=[])
    with pytest.raises(ValidationError):
        JudgeVerdict(tier=Tier.STRONG_FIT, rating=0, evidence=[])
    verdict = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python")],
    )
    result = JudgeResult(candidate_id="c1", **verdict.model_dump())
    assert result.candidate_id == "c1"
    assert result.rating == 8


def test_calibration_output_wraps_list():
    output = CalibrationOutput(
        results=[
            CalibratedResult(
                candidate_id="c1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Top of pool",
            )
        ]
    )
    assert len(output.results) == 1
    assert output.results[0].final_rank == 1


def test_prefilter_result_pass_flag():
    result = PrefilterResult(candidate_id="c1", similarity=0.72, passed=True)
    assert result.passed is True


def test_hallucination_report_all_verified_property():
    verified = HallucinationReport(candidate_id="c1", unverified_quotes=[])
    unverified = HallucinationReport(candidate_id="c1", unverified_quotes=["fabricated quote"])
    assert verified.all_verified is True
    assert unverified.all_verified is False
