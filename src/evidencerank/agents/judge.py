from evidencerank.llm import get_chat_model
from evidencerank.models import CandidateProfile, JudgeResult, JudgeVerdict
from evidencerank.privacy import detect_probable_name, redact_identity

JUDGE_PROMPT = """You are a technical recruiter evaluating a candidate for a role.

## Judging
- Compare the resume directly against the job description below - nothing else.
- A quote must directly demonstrate its claim's specific skill, technology, or responsibility.
- Real work/project experience outweighs a skill merely listed with no context.

## Quoting rules
- Every quote must be copied verbatim: one unbroken span, exactly as it appears in the resume.
- Never join separate lines, bullets, or sections into one quote - not even a job title/date \
header line plus a non-adjacent bullet.
- A long skill/competency list often wraps across several physical lines (e.g. "Core \
Competencies" followed by many items separated by bullets across 2-3 lines). Quote a single \
physical line from it in full, or the section header alone - never combine the header with an \
item from a different physical line, and never skip over items to reach one further down.
- Never submit an empty quote.
- No genuine quote exists for a claim -> omit that claim entirely. Never fabricate a quote, \
explain the gap, or merge text to fill it.

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
