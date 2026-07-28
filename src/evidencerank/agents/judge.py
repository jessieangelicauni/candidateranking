from evidencerank.llm import get_chat_model
from evidencerank.models import CandidateProfile, JDRequirements, JudgeResult, JudgeVerdict
from evidencerank.privacy import detect_probable_name, redact_identity

JUDGE_PROMPT = """You are an experienced technical recruiter evaluating a candidate for a role. \
Reason holistically like a human recruiter: longer relevant experience increases confidence, \
measurable impact matters more than job titles, and technical skill alignment with the role's \
requirements matters most. Give your own holistic judgment — do not compute or describe a \
numeric formula.

Every claim you make MUST be backed by a verbatim quote copied exactly, character-for-character, \
from the "Candidate resume" text block below — and ONLY from that block. Never quote the \
"Candidate structured profile" section (skills/work_history/education/projects): it is \
paraphrased summary data for your background context only, and none of its wording is \
guaranteed to appear in the resume text. For example, quoting "skills: ['TensorFlow']" is NOT \
allowed — that is Python list syntax from the structured profile, not resume text.

Each quote must also directly demonstrate the specific skill, technology, or responsibility \
named in its claim, not merely be true and present somewhere in the resume. For example, if the \
claim is "candidate has machine learning experience," a quote that only establishes years of \
experience in general (e.g. "results-driven engineer with 4+ years of experience") is NOT \
sufficient evidence — the quote must itself name machine learning, a related framework, or a \
related task.

Job requirements:
{jd_requirements}

Candidate resume (identity redacted):
{redacted_cv_text}

Candidate structured profile (background context only — do not quote from this section):
skills: {skills}
work_history: {work_history}
education: {education}
projects: {projects}

Assign a tier (Strong Fit, Moderate Fit, Weak Fit, Not a Fit) and a rating from 1 to 10.
"""


def _build_judge_prompt(jd: JDRequirements, profile: CandidateProfile) -> str:
    contact = profile.contact
    if not contact.name:
        probable_name = detect_probable_name(profile.raw_cv_text)
        if probable_name:
            contact = contact.model_copy(update={"name": probable_name})

    redacted_text = redact_identity(profile.raw_cv_text, contact)

    redacted_work_history = []
    for entry in profile.work_history:
        entry_dump = entry.model_dump()
        entry_dump["achievements"] = [
            redact_identity(achievement, contact) for achievement in entry.achievements
        ]
        redacted_work_history.append(entry_dump)

    redacted_projects = []
    for entry in profile.projects:
        entry_dump = entry.model_dump()
        entry_dump["description"] = redact_identity(entry.description, contact)
        redacted_projects.append(entry_dump)

    return JUDGE_PROMPT.format(
        jd_requirements=jd.model_dump_json(),
        redacted_cv_text=redacted_text,
        skills=profile.skills,
        work_history=redacted_work_history,
        education=[entry.model_dump() for entry in profile.education],
        projects=redacted_projects,
    )


def judge_candidate(jd: JDRequirements, profile: CandidateProfile) -> JudgeResult:
    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompt = _build_judge_prompt(jd, profile)
    verdict = model.invoke(prompt)
    return JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())


def judge_candidates(
    jd: JDRequirements, profiles: list[CandidateProfile], max_concurrency: int
) -> dict[str, JudgeResult]:
    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompts = [_build_judge_prompt(jd, profile) for profile in profiles]
    verdicts = model.batch(prompts, config={"max_concurrency": max_concurrency})
    return {
        profile.candidate_id: JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())
        for profile, verdict in zip(profiles, verdicts)
    }
