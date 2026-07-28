import numpy as np

from evidencerank.agents.prefilter import (
    cosine_similarity,
    prefilter_candidate,
    prefilter_candidates,
)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_prefilter_candidate_matching_skills_passes():
    result = prefilter_candidate(
        candidate_id="c1",
        jd_required_skills=["Python", "Machine Learning", "PyTorch"],
        candidate_skills=["Python", "PyTorch", "Deep Learning", "Model training"],
        threshold=0.4,
    )
    assert result.candidate_id == "c1"
    assert result.passed is True
    assert 0.0 <= result.similarity <= 1.0


def test_prefilter_candidate_unrelated_skills_fails():
    result = prefilter_candidate(
        candidate_id="c2",
        jd_required_skills=["Python", "Machine Learning", "PyTorch"],
        candidate_skills=["Baking", "Pastry decoration", "Kitchen sanitation", "Menu planning"],
        threshold=0.6,
    )
    assert result.passed is False


def test_prefilter_candidates_calls_encode_exactly_once(monkeypatch):
    encode_calls = []

    class FakeEmbedder:
        def encode(self, texts):
            texts = list(texts)
            encode_calls.append(texts)
            return np.array([[1.0, 0.0, 0.0]] * len(texts))

    monkeypatch.setattr("evidencerank.agents.prefilter._get_embedder", lambda: FakeEmbedder())

    results = prefilter_candidates(
        jd_required_skills=["Python", "PyTorch"],
        candidate_skills={"c1": ["Python"], "c2": ["PyTorch"], "c3": ["Baking"]},
        threshold=0.5,
    )

    assert len(encode_calls) == 1
    assert len(encode_calls[0]) == 4  # 1 JD text + 3 candidate texts
    assert set(results.keys()) == {"c1", "c2", "c3"}
    assert all(result.passed for result in results.values())


def test_prefilter_candidates_matches_single_candidate_results():
    jd_required_skills = ["Python", "Machine Learning", "PyTorch"]
    candidate_skills = {
        "matching": ["Python", "PyTorch", "Deep Learning", "Model training"],
        "unrelated": ["Baking", "Pastry decoration", "Kitchen sanitation", "Menu planning"],
    }

    batched = prefilter_candidates(jd_required_skills, candidate_skills, threshold=0.5)

    for candidate_id, skills in candidate_skills.items():
        single = prefilter_candidate(
            candidate_id=candidate_id,
            jd_required_skills=jd_required_skills,
            candidate_skills=skills,
            threshold=0.5,
        )
        assert batched[candidate_id].passed == single.passed
        assert abs(batched[candidate_id].similarity - single.similarity) < 1e-6
