from unittest.mock import MagicMock

from evidencerank.agents.calibrator import calibrate_pool
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
    monkeypatch.setattr(
        "evidencerank.agents.calibrator.get_chat_model",
        lambda stage: fake_chat_model,
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

    results = calibrate_pool(jd, judge_results)

    assert results == expected_output.results
    fake_chat_model.with_structured_output.assert_called_once_with(CalibrationOutput)
