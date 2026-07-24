import re

from evidencerank.models import ContactInfo

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# A phone number requires three digit groups (area/prefix/line, e.g.
# 555-123-4567) separated by delimiters, for a structural minimum of
# 3 + 3 + 4 = 10 digits. A date range like "2015-2020" or "2019 - 2023"
# only has two 4-digit groups (8 digits total) joined by a single
# delimiter, so it can never satisfy this pattern no matter how the
# regex engine backtracks - there simply aren't enough digits to form
# a third group. The optional parens around the area code are captured
# inside the same group as the rest of the match so nothing is left
# dangling in the output.
_PHONE_RE = re.compile(
    r"(?<!\d)(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]\d{4}(?:\s?x\d+)?)(?!\d)"
)


def redact_identity(raw_cv_text: str, contact: ContactInfo) -> str:
    text = raw_cv_text
    if contact.name:
        text = re.sub(re.escape(contact.name), "[REDACTED NAME]", text, flags=re.IGNORECASE)
    if contact.location:
        text = re.sub(re.escape(contact.location), "[REDACTED LOCATION]", text, flags=re.IGNORECASE)
    text = _EMAIL_RE.sub("[REDACTED EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED PHONE]", text)
    return text
