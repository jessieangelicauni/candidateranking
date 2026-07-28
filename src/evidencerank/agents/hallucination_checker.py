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
    """Verify each Judge quote against the candidate's actual resume text.

    Uses profile.raw_cv_text, not the CV-extractor's parsed fields - a genuine,
    verbatim quote is only ever genuine because it's copied from the resume
    itself, so raw_cv_text is the only source that can verify a quote from ANY
    resume section regardless of how (or whether) the extractor captured it.
    Known trade-off: if the extractor mis-transcribes something and the Judge
    echoes that error into a quote, it won't be caught here - but the Judge is
    prompted to quote only from the resume text, not the extracted fields, so
    this only matters if the Judge disobeys that instruction.
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
