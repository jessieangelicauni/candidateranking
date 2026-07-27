from evidencerank.llm import get_chat_model
from evidencerank.models import CandidateProfile, JDRequirements, JudgeResult, JudgeVerdict
from evidencerank.privacy import redact_identity

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


def judge_candidate(jd: JDRequirements, profile: CandidateProfile) -> JudgeResult:
    redacted_text = redact_identity(profile.raw_cv_text, profile.contact)

    redacted_work_history = []
    for entry in profile.work_history:
        entry_dump = entry.model_dump()
        entry_dump["achievements"] = [
            redact_identity(achievement, profile.contact) for achievement in entry.achievements
        ]
        redacted_work_history.append(entry_dump)

    redacted_projects = []
    for entry in profile.projects:
        entry_dump = entry.model_dump()
        entry_dump["description"] = redact_identity(entry.description, profile.contact)
        redacted_projects.append(entry_dump)

    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompt = JUDGE_PROMPT.format(
        jd_requirements=jd.model_dump_json(),
        redacted_cv_text=redacted_text,
        skills=profile.skills,
        work_history=redacted_work_history,
        education=[entry.model_dump() for entry in profile.education],
        projects=redacted_projects,
    )
    verdict = model.invoke(prompt)
    return JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())
