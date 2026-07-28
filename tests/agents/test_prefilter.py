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


def test_prefilter_candidate_passes_when_at_least_two_required_skills_exist():
    # 3 required skills; candidate literally has 2 of them (Python, PyTorch) plus
    # one unrelated skill. 2 matches clears the fixed minimum of 2, so this passes.
    result = prefilter_candidate(
        candidate_id="c1",
        jd_required_skills=["Python", "Machine Learning", "PyTorch"],
        candidate_skills=["Python", "PyTorch", "Baking"],
        threshold=0.9,
    )
    assert result.candidate_id == "c1"
    assert result.passed is True
    assert abs(result.similarity - (2 / 3)) < 1e-6


def test_prefilter_candidate_fails_when_fewer_than_two_required_skills_exist():
    # Candidate only literally has 1 of the 3 required skills (Python) - below
    # the fixed minimum of 2, so this should fail even though Python is an exact match.
    result = prefilter_candidate(
        candidate_id="c2",
        jd_required_skills=["Python", "Machine Learning", "PyTorch"],
        candidate_skills=["Python", "Baking", "Pastry decoration"],
        threshold=0.9,
    )
    assert result.passed is False
    assert abs(result.similarity - (1 / 3)) < 1e-6


def test_prefilter_candidate_passes_on_two_matches_even_when_that_is_not_a_majority():
    # 7 required skills (matching the real JD's shape) - a strict majority would
    # need 4+, but the fixed minimum is 2 regardless of how many requirements
    # there are, so exactly 2 matches out of 7 should still pass.
    result = prefilter_candidate(
        candidate_id="c6",
        jd_required_skills=[
            "Python", "Machine Learning", "Deep Learning", "PyTorch",
            "TensorFlow", "Model training", "Data pipelines",
        ],
        candidate_skills=["Python", "TensorFlow", "Baking"],
        threshold=0.9,
    )
    assert result.passed is True
    assert abs(result.similarity - (2 / 7)) < 1e-6


def test_prefilter_candidate_unrelated_skills_fails():
    result = prefilter_candidate(
        candidate_id="c3",
        jd_required_skills=["Python", "Machine Learning", "PyTorch"],
        candidate_skills=["Baking", "Pastry decoration", "Kitchen sanitation", "Menu planning"],
        threshold=0.6,
    )
    assert result.passed is False
    assert result.similarity == 0.0


def test_prefilter_candidate_handles_no_extracted_skills():
    result = prefilter_candidate(
        candidate_id="c4",
        jd_required_skills=["Python", "Machine Learning", "PyTorch"],
        candidate_skills=[],
        threshold=0.6,
    )
    assert result.passed is False
    assert result.similarity == 0.0


def test_prefilter_candidate_handles_no_required_skills():
    result = prefilter_candidate(
        candidate_id="c5",
        jd_required_skills=[],
        candidate_skills=["Python"],
        threshold=0.6,
    )
    assert result.passed is True
    assert result.similarity == 1.0


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
        candidate_skills={"c1": ["Python"], "c2": ["PyTorch", "TensorFlow"], "c3": []},
        threshold=0.5,
    )

    assert len(encode_calls) == 1
    # 2 required skills + c1's 1 skill + c2's 2 skills + c3's 1 placeholder skill = 6
    assert len(encode_calls[0]) == 6
    assert set(results.keys()) == {"c1", "c2", "c3"}
    # All candidate/required vectors are identical in this fake, so every
    # required skill "exists" for every candidate with real skills.
    assert results["c1"].passed is True
    assert results["c2"].passed is True
    assert results["c3"].passed is False


def test_prefilter_candidates_matches_single_candidate_results():
    jd_required_skills = ["Python", "Machine Learning", "PyTorch"]
    candidate_skills = {
        "matching": ["Python", "PyTorch", "Baking"],
        "unrelated": ["Baking", "Pastry decoration", "Kitchen sanitation", "Menu planning"],
    }

    batched = prefilter_candidates(jd_required_skills, candidate_skills, threshold=0.9)

    for candidate_id, skills in candidate_skills.items():
        single = prefilter_candidate(
            candidate_id=candidate_id,
            jd_required_skills=jd_required_skills,
            candidate_skills=skills,
            threshold=0.9,
        )
        assert batched[candidate_id].passed == single.passed
        assert abs(batched[candidate_id].similarity - single.similarity) < 1e-6
