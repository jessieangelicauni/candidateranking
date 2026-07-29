from unittest.mock import MagicMock

from evidencerank.agents.judge import judge_candidate, judge_candidates
from evidencerank.models import (
    ContactInfo,
    EvidenceClaim,
    CandidateProfile,
    JudgeVerdict,
    Tier,
    WorkHistoryEntry,
)

RAW_JD_TEXT = "We are hiring an ML Engineer. Required skills: Python."


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

    result = judge_candidate(RAW_JD_TEXT, _make_profile())

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


def test_judge_candidate_prompt_uses_raw_jd_text_not_structured_fields(monkeypatch):
    # The Judge now compares the raw job description text directly against
    # the candidate's raw resume text - no pre-parsed JDRequirements JSON or
    # extracted profile fields (skills/work_history/education/projects) are
    # sent, so nothing derived/summarized sits between the LLM and the two
    # source documents.
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
    profile = _make_profile()
    profile.work_history = [
        WorkHistoryEntry(
            title="Engineer", company="Acme Corp", start_date="2019", end_date="2022",
            achievements=["Shipped a recommender system"],
        )
    ]

    judge_candidate(RAW_JD_TEXT, profile)

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert RAW_JD_TEXT in prompt_sent
    assert "required_skills" not in prompt_sent  # no JDRequirements JSON
    assert "work_history:" not in prompt_sent.lower()  # no structured profile block
    assert "Shipped a recommender system" not in prompt_sent  # structured work_history not sent


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
    profile = CandidateProfile(
        candidate_id="c1",
        raw_cv_text="Allison Doyle\nDevOps Engineer\n9 years of DevOps experience",
        contact=ContactInfo(email="allison.doyle@yahoo.com"),
        skills=["DevOps"],
    )

    judge_candidate(RAW_JD_TEXT, profile)

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "Allison Doyle" not in prompt_sent
    assert "[REDACTED NAME]" in prompt_sent


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

    judge_candidate(RAW_JD_TEXT, _make_profile())

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

    judge_candidate(RAW_JD_TEXT, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "no matter how long or short the list is" in prompt_sent
    assert "quoting the skills line itself verbatim" in prompt_sent
    assert "however many items it contains" in prompt_sent


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

    judge_candidate(RAW_JD_TEXT, _make_profile())

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

    judge_candidate(RAW_JD_TEXT, _make_profile())

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

    judge_candidate(RAW_JD_TEXT, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert 'Never quote the "Job requirements" block' in prompt_sent
    assert "it is the role's requirements, not the candidate's resume" in prompt_sent


def test_judge_candidate_prompt_forbids_explanation_in_place_of_missing_quote(monkeypatch):
    # Observed on a real resume (Michael Burton, pure DevOps/SRE work with zero
    # model-training content): instead of omitting a "model training and
    # evaluation" claim as instructed, the Judge submitted an evidence item
    # whose quote was an explanation of why no quote exists ("not directly
    # quoted as the resume does not explicitly mention model training or
    # evaluation tasks") rather than a genuine resume excerpt. The
    # hallucination checker caught and stripped it, but the prompt never
    # explicitly named this failure mode - it only said to omit the claim,
    # not that explaining the omission counts as violating that instruction.
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

    judge_candidate(RAW_JD_TEXT, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "Never write an explanation of why no quote exists AS IF it were the quote itself" in prompt_sent
    assert "not directly quoted as the resume does not explicitly mention model training" in prompt_sent


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

    judge_candidate(RAW_JD_TEXT, _make_profile())

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

    judge_candidate(RAW_JD_TEXT, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "directly demonstrate the specific skill, technology, or responsibility" in prompt_sent
    assert "results-driven engineer with 4+ years of experience" in prompt_sent


def test_judge_candidate_prompt_requires_relevant_quotes_for_negative_claims(monkeypatch):
    # Observed on real resumes (GEval EvidenceRelevancy flagged both): a claim about
    # insufficient ML experience was backed by a quote about Rust experience, and a
    # claim about model training/evaluation was backed by a quote about an unrelated
    # backup/ELK-Stack task that merely happened to contain a percentage figure. The
    # existing claim-relevance rule only had an example for positive ML claims: it
    # needs one for negative/gap claims and for "picked because it has a number in
    # it" quotes too.
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

    judge_candidate(RAW_JD_TEXT, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "applies just as much to negative or gap claims" in prompt_sent
    assert "Junior engineer with 1 years working with Rust" in prompt_sent
    assert "Built automated backup solution using ELK Stack, ensuring 1000% data protection" in prompt_sent
    assert "leave that claim out of the evidence list entirely" in prompt_sent


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

    judge_candidate(RAW_JD_TEXT, _make_profile())

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
    profile_a = _make_profile()
    profile_b = CandidateProfile(
        candidate_id="c2",
        raw_cv_text="Allison Doyle\nDevOps Engineer\n9 years of DevOps experience",
        contact=ContactInfo(name="Allison Doyle"),
        skills=["DevOps"],
    )

    results = judge_candidates(RAW_JD_TEXT, [profile_a, profile_b], max_concurrency=4)

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
