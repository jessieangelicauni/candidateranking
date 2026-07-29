from evidencerank.llm import get_chat_model
from evidencerank.models import JDRequirements

JD_PARSER_PROMPT = """You are an expert technical recruiter extracting structured requirements \
from a job description.

- Extract precisely as stated or clearly implied. Never invent.
- required_skills / nice_to_have_skills: exhaustive, not a sample. Include skills named in \
prose too, not just bullet lists (e.g. "developing backend applications" names Backend \
Development as a skill).
- Split comma-joined bullets into separate entries.
- Mandatory / under "Required" -> required_skills. Preferred / Nice to Have / Bonus -> \
nice_to_have_skills.
- Skills lists = skills and technologies only. Never degrees, even under a "Preferred" heading - \
route degrees to education instead, and only education, never both.
- education: one string, non-empty whenever any degree/field of study is mentioned anywhere in \
the text. Never split into multiple degree/field combinations.
- responsibilities: every distinct duty described, one per item - including duties stated in \
prose with no "Responsibilities" heading.

Job description: {jd_text}
"""


def parse_jd(jd_text: str) -> JDRequirements:
    model = get_chat_model("jd_parser").with_structured_output(JDRequirements)
    return model.invoke(JD_PARSER_PROMPT.format(jd_text=jd_text))
