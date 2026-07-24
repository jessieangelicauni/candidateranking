from unittest.mock import MagicMock

from evidencerank.agents.cv_extractor import extract_cv
from evidencerank.models import ContactInfo, ExtractedProfileFields


def test_extract_cv_assembles_candidate_profile(monkeypatch):
    extracted = ExtractedProfileFields(
        contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
        skills=["Python", "SQL"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = extracted
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    profile = extract_cv("c1", "Jane Doe resume text...")

    assert profile.candidate_id == "c1"
    assert profile.raw_cv_text == "Jane Doe resume text..."
    assert profile.contact.name == "Jane Doe"
    assert profile.skills == ["Python", "SQL"]
    fake_chat_model.with_structured_output.assert_called_once_with(ExtractedProfileFields)
