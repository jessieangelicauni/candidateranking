from unittest.mock import MagicMock

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.models import JDRequirements


def test_parse_jd_returns_structured_requirements(monkeypatch):
    expected = JDRequirements(
        title="Machine Learning Engineer",
        required_skills=["Python", "PyTorch"],
        nice_to_have_skills=["Docker"],
        min_experience_years=2,
        education="Bachelor's in Computer Science",
        responsibilities=["Train models"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = expected
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.jd_parser.get_chat_model",
        lambda stage: fake_chat_model,
    )

    result = parse_jd("Machine Learning Engineer JD text...")

    assert result == expected
    fake_chat_model.with_structured_output.assert_called_once_with(JDRequirements)
    fake_structured_model.invoke.assert_called_once()
