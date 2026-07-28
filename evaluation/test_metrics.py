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


def test_groundedness_metric_scopes_verbatim_check_to_quote_not_claim():
    # A claim is the recruiter's own interpretive synthesis of a quote (e.g. "lacks
    # direct machine learning experience"), not itself a verbatim excerpt. An earlier
    # version of this metric used an implicit `criteria=` string that had the eval
    # judge penalize claims for not having their own separate verbatim match in
    # context, scoring well-reasoned recruiter inferences as "ungrounded" even when
    # every quote was genuinely sourced from the resume. Explicit evaluation_steps
    # (matching the other two metrics) scope the verbatim check to the quote alone.
    steps_text = " ".join(groundedness_metric.evaluation_steps)
    assert "check ONLY the quoted text" in steps_text
    assert "Ignore whether the claim's own wording appears in context" in steps_text
    assert "Do not lower the score because a claim draws a conclusion" in steps_text


def test_recruiter_alignment_metric_uses_input_and_output_params():
    assert LLMTestCaseParams.INPUT in recruiter_alignment_metric.evaluation_params
    assert LLMTestCaseParams.ACTUAL_OUTPUT in recruiter_alignment_metric.evaluation_params
    assert recruiter_alignment_metric.threshold == 0.7


def test_recruiter_alignment_metric_does_not_penalize_unevidenced_requirements():
    steps_text = " ".join(recruiter_alignment_metric.evaluation_steps)
    assert "not a checklist that every requirement must be evidenced" in steps_text
    assert "Do not lower the score merely because the candidate lacks skills" in steps_text


def test_evidence_relevancy_metric_uses_input_and_output_params():
    assert LLMTestCaseParams.INPUT in evidence_relevancy_metric.evaluation_params
    assert LLMTestCaseParams.ACTUAL_OUTPUT in evidence_relevancy_metric.evaluation_params
    assert evidence_relevancy_metric.threshold == 0.7


def test_evidence_relevancy_metric_scopes_check_to_each_claims_own_quote():
    steps_text = " ".join(evidence_relevancy_metric.evaluation_steps)
    assert "not whether the resume as a whole covers every job requirement" in steps_text
    assert "Do not penalize the output for omitting evidence" in steps_text


def test_build_test_case_wraps_fields_correctly():
    case = build_test_case("JD requirements text", "Judge output text", "CV text")

    assert case.input == "JD requirements text"
    assert case.actual_output == "Judge output text"
    assert case.context == ["CV text"]
