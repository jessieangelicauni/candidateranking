import re

from rapidfuzz import fuzz

from evidencerank.models import CandidateProfile, HallucinationReport, JudgeResult

DEFAULT_THRESHOLD = 85.0

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _build_extracted_profile_text(profile: CandidateProfile) -> str:
    """Render the CV-extractor's parsed fields as text to verify Judge quotes against.

    Uses the extracted (skills/work_history/education/projects) fields rather than
    profile.raw_cv_text. Note: this means a quote is "verified" if it matches what
    the extractor parsed, not necessarily what the original resume said - an
    extractor paraphrase or error would pass through as verified.
    """
    parts = [", ".join(profile.skills)]
    for entry in profile.work_history:
        parts.append(f"{entry.title} at {entry.company}")
        parts.extend(entry.achievements)
    for entry in profile.education:
        parts.append(f"{entry.degree}, {entry.institution}")
    for entry in profile.projects:
        parts.append(entry.name)
        parts.append(entry.description)
        if entry.tech:
            parts.append(", ".join(entry.tech))
    return "\n".join(parts)


def check_evidence(
    judge_result: JudgeResult,
    profile: CandidateProfile,
    threshold: float = DEFAULT_THRESHOLD,
) -> HallucinationReport:
    normalized_profile_text = _normalize_whitespace(_build_extracted_profile_text(profile))
    unverified = []
    for claim in judge_result.evidence:
        normalized_quote = _normalize_whitespace(claim.quote)
        score = fuzz.partial_ratio(normalized_quote, normalized_profile_text)
        if score < threshold:
            unverified.append(claim.quote)
    return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=unverified)


def filter_verified_evidence(judge_result: JudgeResult, report: HallucinationReport) -> JudgeResult:
    unverified_quotes = set(report.unverified_quotes)
    verified_evidence = [
        claim for claim in judge_result.evidence if claim.quote not in unverified_quotes
    ]
    return judge_result.model_copy(update={"evidence": verified_evidence})
