import json

from evidencerank.models import (
    CalibratedResult,
    EvidenceClaim,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    Tier,
)
from evidencerank.report import build_json_report, build_markdown_report, write_json_report, write_markdown_report


def _sample_state():
    return {
        "jd": JDRequirements(title="ML Engineer", required_skills=["Python"]),
        "dropped": [{"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}],
        "judge_results": {
            "strong": JudgeResult(
                candidate_id="strong", tier=Tier.STRONG_FIT, rating=9,
                evidence=[EvidenceClaim(claim="Strong Python background", quote="5 years Python")],
            )
        },
        "calibrated_results": [
            CalibratedResult(
                candidate_id="strong", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Only surviving candidate",
            )
        ],
        "hallucination_reports": {
            "strong": HallucinationReport(candidate_id="strong", unverified_quotes=[]),
        },
    }


def test_build_json_report_contains_all_sections():
    report = build_json_report(_sample_state())

    assert report["jd"]["title"] == "ML Engineer"
    assert report["dropped"][0]["candidate_id"] == "weak"
    assert report["judge_results"]["strong"]["rating"] == 9
    assert report["calibrated_results"][0]["final_rank"] == 1
    assert report["hallucination_reports"]["strong"]["unverified_quotes"] == []


def test_write_json_report_writes_valid_json(tmp_path):
    out_path = tmp_path / "report.json"
    write_json_report(_sample_state(), out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["calibrated_results"][0]["candidate_id"] == "strong"


def test_build_markdown_report_has_ranked_table_row():
    markdown = build_markdown_report(_sample_state())

    assert "| Rank | Candidate | Tier | Rating | Calibration Notes |" in markdown
    assert "| 1 | strong | Strong Fit | 9 | Only surviving candidate |" in markdown


def test_write_markdown_report_writes_file(tmp_path):
    out_path = tmp_path / "report.md"
    write_markdown_report(_sample_state(), out_path)

    assert "strong" in out_path.read_text(encoding="utf-8")
