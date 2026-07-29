import re

from rapidfuzz import fuzz

from evidencerank.models import CandidateProfile, HallucinationReport, JudgeResult

DEFAULT_THRESHOLD = 85.0

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def check_evidence(
    judge_result: JudgeResult,
    profile: CandidateProfile,
    threshold: float = DEFAULT_THRESHOLD,
) -> HallucinationReport:
    """Verify every Judge quote against profile.raw_cv_text, the single source of truth. 
    A quote is valid only if it appears verbatim in the original resume. 
    Ignore parsed or extracted CV fields, as they may omit or alter content. 
    If the Judge quotes text containing extraction errors instead of the original resume, mark it as invalid.
    """
    normalized_cv_text = _normalize_whitespace(profile.raw_cv_text)
    unverified = []
    for claim in judge_result.evidence:
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
