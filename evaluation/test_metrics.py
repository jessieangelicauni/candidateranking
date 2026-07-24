from deepeval.test_case import LLMTestCaseParams

from evaluation.metrics import (
    build_test_case,
    evidence_relevancy_metric,
    groundedness_metric,
    recruiter_alignment_metric,
)


def test_groundedness_metric_uses_context_param():
    assert groundedness_metric.threshold == 0.7
    assert LLMTestCaseParams.CONTEXT in groundedness_metric.evaluation_params
    assert LLMTestCaseParams.ACTUAL_OUTPUT in groundedness_metric.evaluation_params


def test_recruiter_alignment_metric_uses_input_and_output_params():
    assert LLMTestCaseParams.INPUT in recruiter_alignment_metric.evaluation_params
    assert LLMTestCaseParams.ACTUAL_OUTPUT in recruiter_alignment_metric.evaluation_params


def test_evidence_relevancy_metric_uses_input_and_output_params():
    assert LLMTestCaseParams.INPUT in evidence_relevancy_metric.evaluation_params
    assert LLMTestCaseParams.ACTUAL_OUTPUT in evidence_relevancy_metric.evaluation_params


def test_build_test_case_wraps_fields_correctly():
    case = build_test_case("JD requirements text", "Judge output text", "CV text")

    assert case.input == "JD requirements text"
    assert case.actual_output == "Judge output text"
    assert case.context == ["CV text"]
