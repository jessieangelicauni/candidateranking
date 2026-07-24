from evidencerank.llm import get_chat_model
from evidencerank.models import JDRequirements

JD_PARSER_PROMPT = """You are an expert technical recruiter. Read the job description below \
and extract its requirements precisely. Do not invent requirements that are not stated or \
clearly implied by the text.

Job description:
{jd_text}
"""


def parse_jd(jd_text: str) -> JDRequirements:
    model = get_chat_model("jd_parser").with_structured_output(JDRequirements)
    return model.invoke(JD_PARSER_PROMPT.format(jd_text=jd_text))
