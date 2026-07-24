from rapidfuzz import fuzz

from evidencerank.models import HallucinationReport, JudgeResult

DEFAULT_THRESHOLD = 85.0


def check_evidence(
    judge_result: JudgeResult,
    raw_cv_text: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> HallucinationReport:
    unverified = []
    for claim in judge_result.evidence:
        score = fuzz.partial_ratio(claim.quote, raw_cv_text)
        if score < threshold:
            unverified.append(claim.quote)
    return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=unverified)
