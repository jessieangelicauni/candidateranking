import json
from pathlib import Path

from evaluation.rank_stability import load_rank_map, rank_stability


def _write_report(path: Path, ranks: dict[str, int]) -> None:
    data = {
        "calibrated_results": [
            {"candidate_id": candidate_id, "final_rank": final_rank}
            for candidate_id, final_rank in ranks.items()
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_rank_map_reads_candidate_ranks(tmp_path):
    report_path = tmp_path / "run1.json"
    _write_report(report_path, {"a": 1, "b": 2})

    rank_map = load_rank_map(report_path)

    assert rank_map == {"a": 1, "b": 2}


def test_rank_stability_identical_rankings_scores_one(tmp_path):
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_report(run1, {"a": 1, "b": 2, "c": 3})
    _write_report(run2, {"a": 1, "b": 2, "c": 3})

    result = rank_stability([str(run1), str(run2)])

    assert result["mean_spearman"] == 1.0
    assert result["mean_kendall_tau"] == 1.0
    assert result["n_runs"] == 2


def test_rank_stability_reversed_rankings_scores_negative_one(tmp_path):
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_report(run1, {"a": 1, "b": 2, "c": 3})
    _write_report(run2, {"a": 3, "b": 2, "c": 1})

    result = rank_stability([str(run1), str(run2)])

    assert result["mean_spearman"] == -1.0
