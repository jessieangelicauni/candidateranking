import re

from rapidfuzz import fuzz

from evidencerank.models import HallucinationReport, JudgeResult

DEFAULT_THRESHOLD = 85.0

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def check_evidence(
    judge_result: JudgeResult,
    raw_cv_text: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> HallucinationReport:
    normalized_cv_text = _normalize_whitespace(raw_cv_text)
    unverified = []
    for claim in judge_result.evidence:
        normalized_quote = _normalize_whitespace(claim.quote)
        score = fuzz.partial_ratio(normalized_quote, normalized_cv_text)
        if score < threshold:
            unverified.append(claim.quote)
    return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=unverified)
