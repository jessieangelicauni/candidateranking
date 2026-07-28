import os

from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

load_dotenv()

_EVAL_MODEL_NAME = os.environ.get("EVIDENCERANK_EVAL_MODEL", "qwen3:14b")
# qwen3:14b is a reasoning model - its chain-of-thought before answering routinely
# exceeds deepeval's ~90s default per-attempt timeout. Raise it unless the caller
# already set their own override.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")
_eval_model = OllamaModel(model=_EVAL_MODEL_NAME)

groundedness_metric = GEval(
    name="Groundedness",
    evaluation_steps=[
        "'actual_output' contains one or more evidence items, each formatted as "
        "'- <claim>: \"<quote>\"'. The claim is the recruiter's own interpretive synthesis "
        "of the quote (an inference, generalization, or negative statement like 'lacks "
        "direct experience in X') - it is not itself required to appear in 'context'.",
        "For EACH evidence item, check ONLY the quoted text (the part inside the double "
        "quotes) against 'context'. Ignore whether the claim's own wording appears in "
        "context - only the quote needs to be grounded there.",
        "Score high if every quote is a genuine, verbatim (or near-verbatim, allowing for "
        "whitespace or line-break differences from PDF extraction) excerpt of 'context'.",
        "Score low only if one or more quotes is fabricated - i.e. it does not appear "
        "anywhere in 'context', or its wording was altered in a way that changes its "
        "meaning from the source text.",
        "Do not lower the score because a claim draws a conclusion, comparison, or "
        "negative assessment that isn't itself a literal quote (e.g. a claim that a "
        "candidate's work history shows infrastructure work rather than machine learning "
        "is a valid interpretation of a quote, not a fabrication, even though that exact "
        "conclusion doesn't appear verbatim in the resume).",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    threshold=0.7,
    model=_eval_model,
)

recruiter_alignment_metric = GEval(
    name="RecruiterAlignment",
    evaluation_steps=[
        "Read 'input' (the job requirements) only as background context for what the role "
        "needs — it is not a checklist that every requirement must be evidenced for the "
        "output to count as sound judgment.",
        "Read 'actual_output': the recruiter's assigned tier, rating, and the evidence "
        "claims backing them.",
        "Judge whether the tier and rating are well-calibrated to the depth and quality of "
        "the evidence actually presented — do experience depth, measurable impact, and "
        "technical skill alignment (for the skills the evidence does cover) plausibly "
        "justify the tier/rating given, rather than superficial keyword matching?",
        "A tier/rating that correctly reflects partial fit (e.g. a moderate rating because "
        "several requirements aren't evidenced) is sound recruiter judgment and should score "
        "high. Do not lower the score merely because the candidate lacks skills the resume "
        "doesn't show — only lower it if the tier/rating is miscalibrated relative to the "
        "evidence shown (e.g. weak evidence rated as Strong Fit, or strong evidence rated "
        "too low).",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=_eval_model,
)

evidence_relevancy_metric = GEval(
    name="EvidenceRelevancy",
    evaluation_steps=[
        "'actual_output' contains one or more evidence items, each an explicit claim "
        "paired with a quote.",
        "For EACH evidence item, judge whether its quote is relevant to and actually "
        "supports the specific claim it is attached to — not whether the resume as a whole "
        "covers every job requirement listed in 'input'.",
        "Do not penalize the output for omitting evidence about a job requirement that no "
        "evidence item claims to address — that is a coverage gap, not an evidence-relevancy "
        "failure, and is out of scope for this metric.",
        "Score high if every included claim's quote genuinely relates to and demonstrates "
        "that claim. Score low if one or more claims pair a quote that does not actually "
        "relate to or demonstrate the named skill/claim (e.g. a quote about a different "
        "technology than the one the claim names).",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=_eval_model,
)


def build_test_case(jd_requirements_text: str, judge_result_text: str, cv_text: str) -> LLMTestCase:
    return LLMTestCase(
        input=jd_requirements_text,
        actual_output=judge_result_text,
        context=[cv_text],
    )
