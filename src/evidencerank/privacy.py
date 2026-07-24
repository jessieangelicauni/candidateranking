import re

from evidencerank.models import ContactInfo

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s()x]{7,}\d)")


def redact_identity(raw_cv_text: str, contact: ContactInfo) -> str:
    text = raw_cv_text
    if contact.name:
        text = text.replace(contact.name, "[REDACTED NAME]")
    if contact.location:
        text = text.replace(contact.location, "[REDACTED LOCATION]")
    text = _EMAIL_RE.sub("[REDACTED EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED PHONE]", text)
    return text
