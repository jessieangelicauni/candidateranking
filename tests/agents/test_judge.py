from unittest.mock import MagicMock

from evidencerank.agents.judge import judge_candidate
from evidencerank.models import (
    ContactInfo,
    EvidenceClaim,
    ExtractedProfileFields,
    CandidateProfile,
    JDRequirements,
    JudgeVerdict,
    ProjectEntry,
    Tier,
    WorkHistoryEntry,
)


def _make_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="c1",
        raw_cv_text=(
            "Daniel Taylor\n"
            "daniel.taylor@protonmail.com | 745-310-7622x683 | Hensleyton, UAE\n"
            "Skills: Python, Machine Learning\n5 years of Python experience"
        ),
        contact=ContactInfo(
            name="Daniel Taylor",
            email="daniel.taylor@protonmail.com",
            phone="745-310-7622x683",
            location="Hensleyton, UAE",
        ),
        skills=["Python", "Machine Learning"],
    )


def test_judge_candidate_redacts_identity_before_prompting(monkeypatch):
    verdict = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = verdict
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.judge.get_chat_model",
        lambda stage: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])

    result = judge_candidate(jd, _make_profile())

    assert result.candidate_id == "c1"
    assert result.tier == Tier.STRONG_FIT
    assert result.rating == 8
    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "Daniel Taylor" not in prompt_sent
    assert "[REDACTED NAME]" in prompt_sent
    assert "daniel.taylor@protonmail.com" not in prompt_sent
    assert "[REDACTED EMAIL]" in prompt_sent
    assert "745-310-7622x683" not in prompt_sent
    assert "[REDACTED PHONE]" in prompt_sent
    assert "Hensleyton, UAE" not in prompt_sent
    assert "[REDACTED LOCATION]" in prompt_sent
    fake_chat_model.with_structured_output.assert_called_once_with(JudgeVerdict)


def test_judge_candidate_redacts_identity_in_structured_free_text_fields(monkeypatch):
    verdict = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = verdict
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.judge.get_chat_model",
        lambda stage: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    profile = _make_profile()
    profile.work_history = [
        WorkHistoryEntry(
            title="Engineer",
            company="Acme Corp",
            start_date="2019",
            end_date="2022",
            achievements=["Contact: Daniel Taylor, daniel.taylor@protonmail.com"],
        )
    ]
    profile.projects = [
        ProjectEntry(
            name="Side Project",
            description="Reach out at daniel.taylor@protonmail.com for details.",
        )
    ]

    judge_candidate(jd, profile)

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "daniel.taylor@protonmail.com" not in prompt_sent
    assert "Contact: [REDACTED NAME], [REDACTED EMAIL]" in prompt_sent
    assert "[REDACTED EMAIL] for details." in prompt_sent


def test_judge_candidate_prompt_scopes_quoting_to_resume_text_only(monkeypatch):
    verdict = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = verdict
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.judge.get_chat_model",
        lambda stage: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])

    judge_candidate(jd, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "ONLY from that block" in prompt_sent
    assert 'Never quote the "Candidate structured profile" section' in prompt_sent
    assert "background context only — do not quote from this section" in prompt_sent


def test_judge_candidate_prompt_requires_claim_relevant_quotes(monkeypatch):
    verdict = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = verdict
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.judge.get_chat_model",
        lambda stage: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])

    judge_candidate(jd, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "directly demonstrate the specific skill, technology, or responsibility" in prompt_sent
    assert "results-driven engineer with 4+ years of experience" in prompt_sent
