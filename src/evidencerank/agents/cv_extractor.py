from evidencerank.llm import get_chat_model
from evidencerank.models import CandidateProfile, ExtractedProfileFields

CV_EXTRACTOR_PROMPT = """You are an expert technical recruiter. Read the resume below and \
extract the candidate's contact info, skills, work history, education, and projects exactly \
as stated. Do not infer skills or experience that are not explicitly present in the text.

Resume:
{cv_text}
"""


def extract_cv(candidate_id: str, cv_text: str) -> CandidateProfile:
    model = get_chat_model("cv_extractor").with_structured_output(ExtractedProfileFields)
    fields = model.invoke(CV_EXTRACTOR_PROMPT.format(cv_text=cv_text))
    return CandidateProfile(
        candidate_id=candidate_id,
        raw_cv_text=cv_text,
        **fields.model_dump(),
    )
