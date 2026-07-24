import numpy as np

from evidencerank.agents.prefilter import cosine_similarity, prefilter_candidate


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
