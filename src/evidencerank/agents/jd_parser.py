from evidencerank.llm import get_chat_model
from evidencerank.models import JDRequirements

JD_PARSER_PROMPT = """You are an expert technical recruiter extracting structured requirements \
from a job description.

- Extract requirements precisely as stated or clearly implied.
- Do not invent requirements the text doesn't state or clearly imply.

Job description: {jd_text}
"""


def parse_jd(jd_text: str) -> JDRequirements:
    model = get_chat_model("jd_parser").with_structured_output(JDRequirements)
    return model.invoke(JD_PARSER_PROMPT.format(jd_text=jd_text))
