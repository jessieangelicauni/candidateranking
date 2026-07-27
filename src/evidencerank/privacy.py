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

# When cv_extractor fails to populate contact.name, there's no known name string
# to substring-match against, but resumes conventionally lead with the
# candidate's name on the first line. This pattern is only checked against that
# first non-blank line, so ordinary body text (e.g. "Skills: Python, SQL") is
# never touched - it requires 2-4 bare Title-Case words with no punctuation,
# which a skills/summary line won't match. This can still misfire on a
# name-shaped title-only header (e.g. "Professional Profile" on its own line),
# but that false positive only costs a decorative header line, versus the
# alternative of a real name leaking into the "blind" Judge prompt - an easy
# trade to make.
_NAME_LIKE_HEADER_RE = re.compile(r"^[A-Z][a-zA-Z'-]*(?:\s+[A-Z][a-zA-Z'-]*){1,3}$")


def detect_probable_name(raw_cv_text: str) -> str | None:
    """Best-effort name guess from a resume's first non-blank line.

    Callers use this once per candidate (when contact.name extraction
    failed) and redact the result everywhere, rather than re-running this
    check against every text fragment being redacted.
    """
    for line in raw_cv_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if _NAME_LIKE_HEADER_RE.fullmatch(stripped) else None
    return None


def redact_identity(raw_cv_text: str, contact: ContactInfo) -> str:
    text = raw_cv_text
    if contact.name:
        text = re.sub(re.escape(contact.name), "[REDACTED NAME]", text, flags=re.IGNORECASE)
    if contact.location:
        text = re.sub(re.escape(contact.location), "[REDACTED LOCATION]", text, flags=re.IGNORECASE)
    text = _EMAIL_RE.sub("[REDACTED EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED PHONE]", text)
    return text
