import numpy as np
from sentence_transformers import SentenceTransformer

from evidencerank.models import PrefilterResult

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def prefilter_candidate(
    candidate_id: str,
    jd_required_skills: list[str],
    candidate_skills: list[str],
    threshold: float = 0.5,
) -> PrefilterResult:
    embedder = _get_embedder()
    jd_text = ", ".join(jd_required_skills)
    candidate_text = ", ".join(candidate_skills)
    jd_vec, candidate_vec = embedder.encode([jd_text, candidate_text])
    similarity = cosine_similarity(jd_vec, candidate_vec)
    return PrefilterResult(
        candidate_id=candidate_id,
        similarity=similarity,
        passed=similarity >= threshold,
    )


def prefilter_candidates(
    jd_required_skills: list[str],
    candidate_skills: dict[str, list[str]],
    threshold: float = 0.5,
) -> dict[str, PrefilterResult]:
    embedder = _get_embedder()
    jd_text = ", ".join(jd_required_skills)
    candidate_ids = list(candidate_skills.keys())
    candidate_texts = [", ".join(candidate_skills[candidate_id]) for candidate_id in candidate_ids]
    vectors = embedder.encode([jd_text, *candidate_texts])
    jd_vec, candidate_vecs = vectors[0], vectors[1:]

    results: dict[str, PrefilterResult] = {}
    for candidate_id, candidate_vec in zip(candidate_ids, candidate_vecs):
        similarity = cosine_similarity(jd_vec, candidate_vec)
        results[candidate_id] = PrefilterResult(
            candidate_id=candidate_id,
            similarity=similarity,
            passed=similarity >= threshold,
        )
    return results
