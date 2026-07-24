from evidencerank.models import ContactInfo
from evidencerank.privacy import redact_identity


def test_redact_identity_removes_name_email_phone_location():
    contact = ContactInfo(
        name="Daniel Taylor",
        email="daniel.taylor@protonmail.com",
        phone="745-310-7622x683",
        location="Hensleyton, UAE",
    )
    raw_text = (
        "Daniel Taylor\n"
        "daniel.taylor@protonmail.com | 745-310-7622x683 | Hensleyton, UAE\n"
        "SUMMARY\nSenior frontend engineer with 7 years of Python experience."
    )

    redacted = redact_identity(raw_text, contact)

    assert "Daniel Taylor" not in redacted
    assert "daniel.taylor@protonmail.com" not in redacted
    assert "745-310-7622x683" not in redacted
    assert "Hensleyton, UAE" not in redacted
    assert "Python experience" in redacted


def test_redact_identity_handles_empty_contact_fields():
    contact = ContactInfo()
    raw_text = "Skills: Python, SQL"

    redacted = redact_identity(raw_text, contact)

    assert redacted == "Skills: Python, SQL"
