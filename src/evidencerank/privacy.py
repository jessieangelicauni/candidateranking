import re

from evidencerank.models import ContactInfo

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# A phone number requires three digit groups (area/prefix/line, e.g.
# 555-123-4567, or a delimiter-free 5551234567) for a structural
# minimum of 3 + 3 + 4 = 10 digits. Both inter-group separators are
# optional so fully delimiter-free numbers match too, but the digit
# counts themselves (\d{3}, \d{3}, \d{4}) are always mandatory - a
# date range like "2015-2020" or "2019 - 2023" only has two 4-digit
# groups (8 digits total), which is not enough digits to ever satisfy
# this pattern regardless of how the (now-optional) separators are
# consumed or how the regex engine backtracks: there simply aren't
# enough digit characters available to fill a third group. The
# optional parens around the area code are captured inside the same
# group as the rest of the match so nothing is left dangling in the
# output, and re.IGNORECASE lets the extension marker match both
# "x" and "X" (e.g. "745-310-7622X683") so no residual fragment like
# "X683" survives redaction.
_PHONE_RE = re.compile(
    r"(?<!\d)(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s?x\d+)?)(?!\d)",
    re.IGNORECASE,
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
