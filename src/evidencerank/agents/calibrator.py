from evidencerank.llm import get_chat_model
from evidencerank.models import CalibratedResult, CalibrationOutput, JDRequirements, JudgeResult

CALIBRATOR_PROMPT = """You are an experienced technical recruiter performing a final calibration \
pass across the full judged candidate pool for one role.

- Each candidate below was already judged independently.
- Your job: reconcile relative ordering across the whole pool — correct for any leniency or \
anchoring drift between the independent judgments — and produce a final rank order (1 = best fit).
- Briefly explain any adjustment you make in calibration_notes.

Job requirements: {jd_requirements}
Independent judge results for every candidate in the pool: {judge_results}
"""

def calibrate_pool(jd: JDRequirements, judge_results: list[JudgeResult]) -> list[CalibratedResult]:
    model = get_chat_model("calibrator").with_structured_output(
        CalibrationOutput
    )
    prompt = CALIBRATOR_PROMPT.format(
        jd_requirements=jd.model_dump_json(),
        judge_results=[result.model_dump() for result in judge_results],
    )
    output = model.invoke(prompt)

    expected_ids = {result.candidate_id for result in judge_results}
    actual_ids = {result.candidate_id for result in output.results}
    
    return output.results
