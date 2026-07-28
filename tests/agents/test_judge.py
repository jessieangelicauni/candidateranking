from unittest.mock import MagicMock

from evidencerank.agents.judge import judge_candidate, judge_candidates
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


def test_judge_candidate_redacts_probable_name_when_extraction_missed_it(monkeypatch):
    # cv_extractor sometimes fails to populate contact.name even though the
    # resume clearly has the candidate's name as its first line. Without a
    # fallback, the real name flows unredacted into the "blind" Judge prompt.
    verdict = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has DevOps experience", quote="9 years of DevOps experience")],
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
    profile = CandidateProfile(
        candidate_id="c1",
        raw_cv_text="Allison Doyle\nDevOps Engineer\n9 years of DevOps experience",
        contact=ContactInfo(email="allison.doyle@yahoo.com"),
        skills=["DevOps"],
        work_history=[
            WorkHistoryEntry(
                title="DevOps Engineer",
                company="Acme Corp",
                start_date="2019",
                end_date="2022",
                achievements=["Reported directly to Allison Doyle's team lead"],
            )
        ],
    )

    judge_candidate(jd, profile)

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "Allison Doyle" not in prompt_sent
    assert prompt_sent.count("[REDACTED NAME]") == 2


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


def test_judge_candidate_prompt_forbids_quoting_full_structured_lists(monkeypatch):
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
    assert "no matter how long or short the list is" in prompt_sent
    assert "neither is quoting the skills line itself verbatim" in prompt_sent
    assert "however many items it contains" in prompt_sent


def test_judge_candidate_prompt_renders_skills_as_plain_line_not_python_list(monkeypatch):
    # Observed on a real resume (cv_00019) whose extracted skills list, once
    # comma-joined, is byte-for-byte identical to the resume's own skills
    # line - the Judge kept quoting the structured profile's "skills: [...]"
    # Python-list rendering as if it were a resume quote, and repeated prose
    # warnings didn't stop it. Rendering skills as a plain comma-separated
    # line (matching how the resume itself lists skills) removes the
    # bracket/quote syntax the model was copy-pasting as a "quote".
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
    assert "skills: Python, Machine Learning" in prompt_sent
    assert "skills: ['Python', 'Machine Learning']" not in prompt_sent


def test_judge_candidate_prompt_forbids_quoting_paraphrased_prose_fields(monkeypatch):
    # work_history/education/projects are rendered as Python list/dict
    # syntax, an obvious tell that they aren't resume text. education.degree,
    # by contrast, is a plain human-readable string (e.g. "Master's Degree,
    # Computer Science")
    # that the CV-extractor may have paraphrased from the resume's actual
    # wording (e.g. "M.Sc. in Computer Science") - the prompt must warn
    # against echoing it as if it were a genuine quote too.
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
    assert "education" in prompt_sent.lower()
    assert "M.Sc." in prompt_sent
    assert "Master's Degree, Computer Science" in prompt_sent


def test_judge_candidate_prompt_forbids_recombining_structured_fields_into_new_sentences(monkeypatch):
    # Observed on a real resume: the structured education field stores
    # degree/institution/year separately, and the Judge assembled them into
    # a new sentence ("B.Sc. in Artificial Intelligence, TU Delft 2019") in a
    # different order than the resume actually lays them out. That's not a
    # verbatim quote even though every underlying fact is real - the prompt
    # must forbid synthesizing sentences from combined fields, not just
    # quoting a single field verbatim.
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
    assert "combining two or more separate structured fields into a new sentence" in prompt_sent
    assert "B.Sc. in Artificial Intelligence, TU Delft 2019" in prompt_sent


def test_judge_candidate_prompt_forbids_quoting_job_requirements(monkeypatch):
    # Observed on a real resume with zero ML content: the Judge produced an
    # evidence item quoting "Machine Learning" verbatim from the JD's
    # required_skills list, not from the candidate's resume at all - a
    # fabrication the prompt never explicitly warned against, since the
    # existing rules only addressed the structured-profile block.
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
    assert 'Never quote the "Job requirements" block' in prompt_sent
    assert "it is the role's requirements, not the candidate's resume" in prompt_sent


def test_judge_candidate_prompt_requires_contiguous_non_empty_quotes(monkeypatch):
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
    assert "never join two or more separate lines, bullets, or sections together" in prompt_sent
    assert "Never submit an empty or blank quote" in prompt_sent


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


def test_judge_candidate_prompt_weighs_applied_experience_over_skill_list_mentions(monkeypatch):
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
    assert "weak evidence of genuine proficiency" in prompt_sent
    assert "counts far more heavily toward the rating than the same skill merely being listed" in prompt_sent


def test_judge_candidates_batches_prompts_and_returns_results_by_candidate_id(monkeypatch):
    verdict_a = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    verdict_b = JudgeVerdict(
        tier=Tier.WEAK_FIT,
        rating=3,
        evidence=[EvidenceClaim(claim="Has DevOps experience", quote="9 years of DevOps experience")],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.batch.return_value = [verdict_a, verdict_b]
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.judge.get_chat_model",
        lambda stage: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    profile_a = _make_profile()
    profile_b = CandidateProfile(
        candidate_id="c2",
        raw_cv_text="Allison Doyle\nDevOps Engineer\n9 years of DevOps experience",
        contact=ContactInfo(name="Allison Doyle"),
        skills=["DevOps"],
    )

    results = judge_candidates(jd, [profile_a, profile_b], max_concurrency=4)

    assert set(results.keys()) == {"c1", "c2"}
    assert results["c1"].candidate_id == "c1"
    assert results["c1"].tier == Tier.STRONG_FIT
    assert results["c1"].rating == 8
    assert results["c2"].candidate_id == "c2"
    assert results["c2"].tier == Tier.WEAK_FIT
    assert results["c2"].rating == 3
    call_args, call_kwargs = fake_structured_model.batch.call_args
    prompts_sent = call_args[0]
    assert len(prompts_sent) == 2
    assert "Allison Doyle" not in prompts_sent[1]
    assert call_kwargs["config"] == {"max_concurrency": 4}
