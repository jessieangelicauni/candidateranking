import time
from typing import TypedDict

import click
from langgraph.graph import END, StateGraph

from evidencerank.agents.calibrator import calibrate_pool
from evidencerank.agents.cv_extractor import cached_extract_cv
from evidencerank.agents.hallucination_checker import (
    DEFAULT_THRESHOLD,
    check_evidence,
    filter_verified_evidence,
)
from evidencerank.agents.judge import judge_candidate
from evidencerank.agents.prefilter import prefilter_candidate
from evidencerank.agents.shortlist import select_shortlist
from evidencerank.models import (
    CalibratedResult,
    CandidateProfile,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    PrefilterResult,
)


class PipelineState(TypedDict, total=False):
    jd: JDRequirements
    raw_resumes: dict[str, str]
    profiles: dict[str, CandidateProfile]
    prefilter_results: dict[str, PrefilterResult]
    dropped: list[dict[str, str]]
    judge_results: dict[str, JudgeResult]
    shortlisted_results: dict[str, JudgeResult]
    not_shortlisted: list[dict[str, str]]
    calibrated_results: list[CalibratedResult]
    hallucination_reports: dict[str, HallucinationReport]
    stage_timings: dict[str, float]
    prefilter_threshold: float
    hallucination_threshold: float


def extract_profiles_node(state: PipelineState) -> dict:
    click.echo("Running stage: extract_profiles")
    profiles = {
        candidate_id: cached_extract_cv(candidate_id, raw_text)
        for candidate_id, raw_text in state["raw_resumes"].items()
    }
    return {"profiles": profiles}


def prefilter_node(state: PipelineState) -> dict:
    click.echo("Running stage: prefilter")
    threshold = state.get("prefilter_threshold", 0.5)
    results: dict[str, PrefilterResult] = {}
    dropped: list[dict[str, str]] = []
    for candidate_id, profile in state["profiles"].items():
        result = prefilter_candidate(
            candidate_id,
            state["jd"].required_skills,
            profile.skills,
            threshold=threshold,
        )
        results[candidate_id] = result
        if not result.passed:
            dropped.append(
                {"candidate_id": candidate_id, "reason": "pre-filter: no relevant skill overlap"}
            )
    return {"prefilter_results": results, "dropped": dropped}


def judge_node(state: PipelineState) -> dict:
    click.echo("Running stage: judge")
    judge_results: dict[str, JudgeResult] = {}
    for candidate_id, result in state["prefilter_results"].items():
        if not result.passed:
            continue
        profile = state["profiles"][candidate_id]
        judge_results[candidate_id] = judge_candidate(state["jd"], profile)
    return {"judge_results": judge_results}


def shortlist_node(state: PipelineState) -> dict:
    click.echo("Running stage: shortlist")
    shortlisted, not_shortlisted = select_shortlist(list(state["judge_results"].values()))
    return {
        "shortlisted_results": {result.candidate_id: result for result in shortlisted},
        "not_shortlisted": not_shortlisted,
    }


def calibrate_node(state: PipelineState) -> dict:
    click.echo("Running stage: calibrate")
    calibrated = calibrate_pool(state["jd"], list(state["shortlisted_results"].values()))
    return {"calibrated_results": calibrated}


def hallucination_check_node(state: PipelineState) -> dict:
    click.echo("Running stage: hallucination_check")
    threshold = state.get("hallucination_threshold", DEFAULT_THRESHOLD)
    reports = {}
    filtered_judge_results = {}
    for candidate_id, judge_result in state["judge_results"].items():
        raw_text = state["profiles"][candidate_id].raw_cv_text
        report = check_evidence(judge_result, raw_text, threshold=threshold)
        reports[candidate_id] = report
        filtered_judge_results[candidate_id] = filter_verified_evidence(judge_result, report)
    return {"hallucination_reports": reports, "judge_results": filtered_judge_results}


def _timed_node(name, node_fn):
    def wrapped(state: PipelineState) -> dict:
        start = time.perf_counter()
        result = dict(node_fn(state))
        elapsed = time.perf_counter() - start
        timings = dict(state.get("stage_timings", {}))
        timings[name] = elapsed
        result["stage_timings"] = timings
        return result
    return wrapped


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extract_profiles", _timed_node("extract_profiles", extract_profiles_node))
    graph.add_node("prefilter", _timed_node("prefilter", prefilter_node))
    graph.add_node("judge", _timed_node("judge", judge_node))
    graph.add_node("hallucination_check", _timed_node("hallucination_check", hallucination_check_node))
    graph.add_node("shortlist", _timed_node("shortlist", shortlist_node))
    graph.add_node("calibrate", _timed_node("calibrate", calibrate_node))

    graph.set_entry_point("extract_profiles")
    graph.add_edge("extract_profiles", "prefilter")
    graph.add_edge("prefilter", "judge")
    graph.add_edge("judge", "hallucination_check")
    graph.add_edge("hallucination_check", "shortlist")
    graph.add_edge("shortlist", "calibrate")
    graph.add_edge("calibrate", END)

    return graph.compile()
