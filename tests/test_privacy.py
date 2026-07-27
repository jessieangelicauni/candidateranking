from evidencerank.models import ContactInfo
from evidencerank.privacy import detect_probable_name, redact_identity


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


def test_detect_probable_name_finds_name_shaped_first_line():
    raw_text = (
        "Allison Doyle\n"
        "DevOps / SRE / MLOps · 9.2 years · Lake Jamesmouth, Japan\n"
        "allison.doyle@yahoo.com · 563-542-1607x33754\n"
        "Professional Profile\nSenior technical leader specializing in devops."
    )

    assert detect_probable_name(raw_text) == "Allison Doyle"


def test_detect_probable_name_returns_none_for_non_name_first_line():
    assert detect_probable_name("Skills: Python, SQL") is None


def test_detect_probable_name_skips_leading_blank_lines():
    assert detect_probable_name("\n\n  Allison Doyle  \nDevOps Engineer") == "Allison Doyle"


def test_redact_identity_redacts_dot_formatted_phone():
    contact = ContactInfo(name="Jane Doe", phone="555.123.4567")
    raw_text = "Contact: 555.123.4567"

    redacted = redact_identity(raw_text, contact)

    assert "555.123.4567" not in redacted
    assert "[REDACTED PHONE]" in redacted


def test_redact_identity_preserves_date_ranges_but_redacts_real_phone():
    contact = ContactInfo(name="Jane Doe", phone="555-123-4567")
    raw_text = (
        "WORK HISTORY\n"
        "Senior Engineer, Acme Corp, 2015-2020\n"
        "Junior Engineer, Beta Inc, 2019 - 2023\n"
        "Phone: 555-123-4567"
    )

    redacted = redact_identity(raw_text, contact)

    assert "2015-2020" in redacted
    assert "2019 - 2023" in redacted
    assert "555-123-4567" not in redacted
    assert "[REDACTED PHONE]" in redacted


def test_redact_identity_fully_redacts_parenthesized_phone_no_leftover_chars():
    contact = ContactInfo(name="Jane Doe", phone="(555) 123-4567")
    raw_text = "Call me at (555) 123-4567 anytime."

    redacted = redact_identity(raw_text, contact)

    assert redacted == "Call me at [REDACTED PHONE] anytime."


def test_redact_identity_is_case_insensitive_for_name():
    contact = ContactInfo(name="Daniel Taylor")
    raw_text = "DANIEL TAYLOR\nSUMMARY\nSenior engineer."

    redacted = redact_identity(raw_text, contact)

    assert "DANIEL TAYLOR" not in redacted
    assert "[REDACTED NAME]" in redacted


def test_redact_identity_redacts_delimiter_free_phone():
    contact = ContactInfo(name="Jane Doe", phone="5551234567")
    raw_text = "Phone: 5551234567"

    redacted = redact_identity(raw_text, contact)

    assert "5551234567" not in redacted
    assert redacted == "Phone: [REDACTED PHONE]"


def test_redact_identity_fully_redacts_capital_x_extension_no_leftover_chars():
    contact = ContactInfo(name="Jane Doe", phone="745-310-7622X683")
    raw_text = "Call 745-310-7622X683 for details."

    redacted = redact_identity(raw_text, contact)

    assert "745-310-7622X683" not in redacted
    assert "X683" not in redacted
    assert redacted == "Call [REDACTED PHONE] for details."


def test_redact_identity_still_preserves_date_ranges_after_optional_separator_fix():
    contact = ContactInfo(name="Jane Doe")
    raw_text = (
        "WORK HISTORY\n"
        "Senior Engineer, Acme Corp, 2015-2020\n"
        "Junior Engineer, Beta Inc, 2019 - 2023"
    )

    redacted = redact_identity(raw_text, contact)

    assert "2015-2020" in redacted
    assert "2019 - 2023" in redacted
    assert "[REDACTED PHONE]" not in redacted
