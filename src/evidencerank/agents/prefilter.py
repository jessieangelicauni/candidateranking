import numpy as np
from sentence_transformers import SentenceTransformer

from evidencerank.models import PrefilterResult

MIN_REQUIRED_SKILLS_MATCHED = 2

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _required_skill_coverage(
    req_vecs: np.ndarray, skill_vecs: np.ndarray, threshold: float
) -> tuple[int, float]:
    """Task: Count how many required skills are matched by the candidate. 
    A required skill is considered matched if at least one candidate skill has a cosine similarity greater than the threshold. 
    Return both the matched count and the match fraction.

    Limitation: Short-phrase embeddings can miss semantically related skills expressed with different terminology (e.g., "LangChain" or "MLOps" vs. "Machine Learning"), so some relevant matches may fall below the threshold. 
    This approach prioritizes reducing false positives over maximizing recall.
    """
    matched = sum(
        max(cosine_similarity(req_vec, skill_vec) for skill_vec in skill_vecs) >= threshold
        for req_vec in req_vecs
    )
    return matched, matched / len(req_vecs)


def prefilter_candidate(
    candidate_id: str,
    jd_required_skills: list[str],
    candidate_skills: list[str],
    threshold: float = 0.7,
) -> PrefilterResult:
    if not jd_required_skills:
        return PrefilterResult(candidate_id=candidate_id, similarity=1.0, passed=True)

    embedder = _get_embedder()
    req_vecs = embedder.encode(jd_required_skills)
    skill_vecs = embedder.encode(candidate_skills) if candidate_skills else None

    if skill_vecs is None:
        matched, fraction = 0, 0.0
    else:
        matched, fraction = _required_skill_coverage(req_vecs, skill_vecs, threshold)

    return PrefilterResult(
        candidate_id=candidate_id,
        similarity=fraction,
        passed=matched >= min(MIN_REQUIRED_SKILLS_MATCHED, len(jd_required_skills)),
    )


def prefilter_candidates(
    jd_required_skills: list[str],
    candidate_skills: dict[str, list[str]],
    threshold: float = 0.7,
) -> dict[str, PrefilterResult]:
    if not jd_required_skills:
        return {
            candidate_id: PrefilterResult(candidate_id=candidate_id, similarity=1.0, passed=True)
            for candidate_id in candidate_skills
        }

    embedder = _get_embedder()
    candidate_ids = list(candidate_skills.keys())

    all_texts = list(jd_required_skills)
    skill_spans: dict[str, tuple[int, int]] = {}
    for candidate_id in candidate_ids:
        skills = candidate_skills[candidate_id] or ["(no skills extracted)"]
        start = len(all_texts)
        all_texts.extend(skills)
        skill_spans[candidate_id] = (start, len(all_texts))

    vectors = embedder.encode(all_texts)
    n_required = len(jd_required_skills)
    req_vecs = vectors[:n_required]

    results: dict[str, PrefilterResult] = {}
    for candidate_id in candidate_ids:
        start, end = skill_spans[candidate_id]
        has_real_skills = bool(candidate_skills[candidate_id])
        if has_real_skills:
            matched, fraction = _required_skill_coverage(req_vecs, vectors[start:end], threshold)
        else:
            matched, fraction = 0, 0.0
        results[candidate_id] = PrefilterResult(
            candidate_id=candidate_id,
            similarity=fraction,
            passed=matched >= min(MIN_REQUIRED_SKILLS_MATCHED, n_required),
        )
    return results
