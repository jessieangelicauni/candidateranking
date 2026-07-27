from typing import TypedDict

import click
from langgraph.graph import END, StateGraph

from evidencerank.agents.calibrator import calibrate_pool
from evidencerank.agents.cv_extractor import cached_extract_cv
from evidencerank.agents.hallucination_checker import DEFAULT_THRESHOLD, check_evidence
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
    shortlisted_ids: set[str]
    not_shortlisted: list[dict[str, str]]
    calibrated_results: list[CalibratedResult]
    hallucination_reports: dict[str, HallucinationReport]
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
        "shortlisted_ids": {result.candidate_id for result in shortlisted},
        "not_shortlisted": not_shortlisted,
    }


def calibrate_node(state: PipelineState) -> dict:
    click.echo("Running stage: calibrate")
    pool = [
        result
        for result in state["judge_results"].values()
        if result.candidate_id in state["shortlisted_ids"]
    ]
    calibrated = calibrate_pool(state["jd"], pool)
    return {"calibrated_results": calibrated}


def hallucination_check_node(state: PipelineState) -> dict:
    click.echo("Running stage: hallucination_check")
    threshold = state.get("hallucination_threshold", DEFAULT_THRESHOLD)
    reports = {}
    for candidate_id, judge_result in state["judge_results"].items():
        raw_text = state["profiles"][candidate_id].raw_cv_text
        reports[candidate_id] = check_evidence(judge_result, raw_text, threshold=threshold)
    return {"hallucination_reports": reports}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extract_profiles", extract_profiles_node)
    graph.add_node("prefilter", prefilter_node)
    graph.add_node("judge", judge_node)
    graph.add_node("shortlist", shortlist_node)
    graph.add_node("calibrate", calibrate_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    graph.set_entry_point("extract_profiles")
    graph.add_edge("extract_profiles", "prefilter")
    graph.add_edge("prefilter", "judge")
    graph.add_edge("judge", "shortlist")
    graph.add_edge("shortlist", "calibrate")
    graph.add_edge("calibrate", "hallucination_check")
    graph.add_edge("hallucination_check", END)

    return graph.compile()
