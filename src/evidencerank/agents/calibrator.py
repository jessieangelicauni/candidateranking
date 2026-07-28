from evidencerank.llm import get_chat_model
from evidencerank.models import CalibratedResult, CalibrationOutput, JDRequirements, JudgeResult

CALIBRATOR_PROMPT = """You are an experienced technical recruiter performing a final calibration \
pass across the full judged candidate pool for one role. Each candidate below was already judged \
independently; your job is to reconcile relative ordering across the whole pool — correct for \
any leniency or anchoring drift between the independent judgments — and produce a final rank \
order (1 = best fit). Briefly explain any adjustment you make in calibration_notes.

Job requirements:
{jd_requirements}

Independent judge results for every candidate in the pool:
{judge_results}
"""

# The calibrator prompt scales with pool size (every judge result is embedded in one
# call). Ollama defaults to a 4096-token runtime context when num_ctx isn't set, which
# silently truncates the prompt for pools beyond a handful of candidates and drops
# candidates from the response with no error. 32768 matches the max context of the
# models this stage is configured to use by default.

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
