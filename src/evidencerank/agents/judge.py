from evidencerank.llm import get_chat_model
from evidencerank.models import CandidateProfile, JDRequirements, JudgeResult, JudgeVerdict
from evidencerank.privacy import redact_identity

JUDGE_PROMPT = """You are an experienced technical recruiter evaluating a candidate for a role. \
Reason holistically like a human recruiter: longer relevant experience increases confidence, \
measurable impact matters more than job titles, and technical skill alignment with the role's \
requirements matters most. Give your own holistic judgment — do not compute or describe a \
numeric formula.

Every claim you make MUST be backed by a verbatim quote copied exactly from the resume text \
below. Never quote text that does not appear in the resume text.

Job requirements:
{jd_requirements}

Candidate resume (identity redacted):
{redacted_cv_text}

Candidate structured profile:
skills: {skills}
work_history: {work_history}
education: {education}
projects: {projects}

Assign a tier (Strong Fit, Moderate Fit, Weak Fit, Not a Fit) and a rating from 1 to 10.
"""


def judge_candidate(jd: JDRequirements, profile: CandidateProfile) -> JudgeResult:
    redacted_text = redact_identity(profile.raw_cv_text, profile.contact)
    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompt = JUDGE_PROMPT.format(
        jd_requirements=jd.model_dump_json(),
        redacted_cv_text=redacted_text,
        skills=profile.skills,
        work_history=[entry.model_dump() for entry in profile.work_history],
        education=[entry.model_dump() for entry in profile.education],
        projects=[entry.model_dump() for entry in profile.projects],
    )
    verdict = model.invoke(prompt)
    return JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())
