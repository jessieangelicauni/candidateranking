import re

from rapidfuzz import fuzz

from evidencerank.models import CandidateProfile, HallucinationReport, JudgeResult

DEFAULT_THRESHOLD = 85.0

_WHITESPACE_RE = re.compile(r"\s+")
_ELLIPSIS_MARKERS = ("...", "…")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def check_evidence(
    judge_result: JudgeResult,
    profile: CandidateProfile,
    threshold: float = DEFAULT_THRESHOLD,
) -> HallucinationReport:
    normalized_cv_text = _normalize_whitespace(profile.raw_cv_text)
    unverified = []
    for claim in judge_result.evidence:
        if any(marker in claim.quote for marker in _ELLIPSIS_MARKERS):
            unverified.append(claim.quote)
            continue
        normalized_quote = _normalize_whitespace(claim.quote)
        score = fuzz.partial_ratio(normalized_quote, normalized_cv_text)
        if score < threshold:
            unverified.append(claim.quote)
    return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=unverified)


def filter_verified_evidence(judge_result: JudgeResult, report: HallucinationReport) -> JudgeResult:
    unverified_quotes = set(report.unverified_quotes)
    verified_evidence = [
        claim for claim in judge_result.evidence if claim.quote not in unverified_quotes
    ]
    return judge_result.model_copy(update={"evidence": verified_evidence})
