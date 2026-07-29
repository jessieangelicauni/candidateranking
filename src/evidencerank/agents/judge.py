from evidencerank.llm import get_chat_model
from evidencerank.models import CandidateProfile, JudgeResult, JudgeVerdict
from evidencerank.privacy import detect_probable_name, redact_identity

JUDGE_PROMPT = """You are an experienced technical recruiter evaluating a candidate for a role.

## How to judge
- Compare the candidate's resume directly against the job description below - judge fit \
purely from what the resume itself says, not from any pre-parsed summary of either document.
- Each quote must also directly demonstrate the specific skill, technology, or responsibility \
named in its claim, not merely be true and present somewhere in the resume.
- Reason holistically like a human recruiter: longer relevant experience increases confidence, \
measurable impact matters more than job titles, and technical skill alignment with the role's \
requirements matters most.
- Give greater weight to skills demonstrated through real work or project experience than skills listed without context.
A skill mentioned only in a skills list is weak evidence of proficiency, while a skill applied to project is strong evidence.

## Quoting rules — critical, read all of these
- Every claim you make MUST be backed by a verbatim quote copied exactly from the "Candidate resume" text block.
- Find and quote the resume's own line instead, wherever the skill is actually demonstrated.
- Never submit an empty or blank quote.
- If you cannot find a genuine verbatim quote to support a claim, do not include that claim as \
an evidence item at all.** This is the one correct response every time evidence is missing — \
never fabricate a quote, never explain the omission, never merge unrelated text to fill the gap.

Job description: {jd_text}
Candidate resume (identity redacted): {redacted_cv_text}

Assign a tier (Strong Fit, Moderate Fit, Weak Fit, Not a Fit) and a rating from 1 to 10.
"""


def _build_judge_prompt(raw_jd_text: str, profile: CandidateProfile) -> str:
    contact = profile.contact
    if not contact.name:
        probable_name = detect_probable_name(profile.raw_cv_text)
        if probable_name:
            contact = contact.model_copy(update={"name": probable_name})

    redacted_text = redact_identity(profile.raw_cv_text, contact)

    return JUDGE_PROMPT.format(jd_text=raw_jd_text, redacted_cv_text=redacted_text)


def judge_candidate(raw_jd_text: str, profile: CandidateProfile) -> JudgeResult:
    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompt = _build_judge_prompt(raw_jd_text, profile)
    verdict = model.invoke(prompt)
    return JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())


def judge_candidates(
    raw_jd_text: str, profiles: list[CandidateProfile], max_concurrency: int
) -> dict[str, JudgeResult]:
    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompts = [_build_judge_prompt(raw_jd_text, profile) for profile in profiles]
    verdicts = model.batch(prompts, config={"max_concurrency": max_concurrency})
    return {
        profile.candidate_id: JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())
        for profile, verdict in zip(profiles, verdicts)
    }
