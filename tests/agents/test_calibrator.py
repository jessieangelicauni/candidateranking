from unittest.mock import MagicMock

import pytest

from evidencerank.agents.calibrator import CALIBRATOR_NUM_CTX, calibrate_pool
from evidencerank.models import (
    CalibratedResult,
    CalibrationOutput,
    EvidenceClaim,
    JDRequirements,
    JudgeResult,
    Tier,
)


def test_calibrate_pool_returns_ranked_list(monkeypatch):
    expected_output = CalibrationOutput(
        results=[
            CalibratedResult(
                candidate_id="c1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Deepest relevant experience in pool",
            ),
            CalibratedResult(
                candidate_id="c2", final_rank=2, tier=Tier.MODERATE_FIT,
                rating=6, calibration_notes="Adjacent domain experience only",
            ),
        ]
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = expected_output
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    fake_get_chat_model = MagicMock(return_value=fake_chat_model)
    monkeypatch.setattr("evidencerank.agents.calibrator.get_chat_model", fake_get_chat_model)
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    judge_results = [
        JudgeResult(
            candidate_id="c1", tier=Tier.STRONG_FIT, rating=9,
            evidence=[EvidenceClaim(claim="Strong Python background", quote="5 years Python")],
        ),
        JudgeResult(
            candidate_id="c2", tier=Tier.MODERATE_FIT, rating=6,
            evidence=[EvidenceClaim(claim="Some ML exposure", quote="1 year of ML projects")],
        ),
    ]

    results = calibrate_pool(jd, judge_results)

    assert results == expected_output.results
    fake_get_chat_model.assert_called_once_with("calibrator", num_ctx=CALIBRATOR_NUM_CTX)
    fake_chat_model.with_structured_output.assert_called_once_with(CalibrationOutput)


def test_calibrate_pool_raises_when_candidates_missing(monkeypatch):
    incomplete_output = CalibrationOutput(
        results=[
            CalibratedResult(
                candidate_id="c1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Deepest relevant experience in pool",
            ),
        ]
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = incomplete_output
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.calibrator.get_chat_model",
        lambda stage, **kwargs: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    judge_results = [
        JudgeResult(
            candidate_id="c1", tier=Tier.STRONG_FIT, rating=9,
            evidence=[EvidenceClaim(claim="Strong Python background", quote="5 years Python")],
        ),
        JudgeResult(
            candidate_id="c2", tier=Tier.MODERATE_FIT, rating=6,
            evidence=[EvidenceClaim(claim="Some ML exposure", quote="1 year of ML projects")],
        ),
    ]

    with pytest.raises(ValueError, match="c2"):
        calibrate_pool(jd, judge_results)
