from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

groundedness_metric = GEval(
    name="Groundedness",
    criteria=(
        "Determine whether every claim in 'actual_output' is directly supported by a "
        "verbatim quote that appears in 'context'. Penalize any claim not backed by a "
        "quote found in the context."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    threshold=0.7,
)

recruiter_alignment_metric = GEval(
    name="RecruiterAlignment",
    criteria=(
        "Determine whether 'actual_output' reflects sound recruiter judgment given "
        "'input' (the job requirements): does it weigh relevant experience depth, "
        "measurable impact, and technical skill alignment appropriately, rather than "
        "superficial keyword matching?"
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

evidence_relevancy_metric = GEval(
    name="EvidenceRelevancy",
    criteria=(
        "Determine whether the quoted evidence in 'actual_output' is relevant to the "
        "job requirement claim it supports in 'input', not merely present somewhere in "
        "the resume."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)


def build_test_case(jd_requirements_text: str, judge_result_text: str, cv_text: str) -> LLMTestCase:
    return LLMTestCase(
        input=jd_requirements_text,
        actual_output=judge_result_text,
        context=[cv_text],
    )
