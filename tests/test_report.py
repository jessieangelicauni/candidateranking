import json

from evidencerank.models import (
    CalibratedResult,
    CandidateProfile,
    ContactInfo,
    EvidenceClaim,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    PrefilterResult,
    Tier,
)
from evidencerank.report import build_json_report, build_markdown_report, write_json_report, write_markdown_report


def _sample_state():
    return {
        "jd": JDRequirements(title="ML Engineer", required_skills=["Python"]),
        "profiles": {
            "strong": CandidateProfile(
                candidate_id="strong",
                raw_cv_text="Jane Doe, 5 years Python",
                contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
                skills=["Python"],
            )
        },
        "prefilter_results": {
            "strong": PrefilterResult(candidate_id="strong", similarity=0.9, passed=True),
            "weak": PrefilterResult(candidate_id="weak", similarity=0.1, passed=False),
        },
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
    assert report["profiles"]["strong"]["candidate_id"] == "strong"
    assert report["profiles"]["strong"]["raw_cv_text"] == "Jane Doe, 5 years Python"
    assert report["profiles"]["strong"]["contact"]["name"] == "Jane Doe"
    assert report["prefilter_results"]["strong"]["similarity"] == 0.9
    assert report["prefilter_results"]["strong"]["passed"] is True
    assert report["prefilter_results"]["weak"]["passed"] is False
    assert report["dropped"][0]["candidate_id"] == "weak"
    assert report["judge_results"]["strong"]["rating"] == 9
    assert report["calibrated_results"][0]["final_rank"] == 1
    assert report["hallucination_reports"]["strong"]["unverified_quotes"] == []


def test_build_json_report_defaults_missing_stages_to_empty():
    minimal_state = {"jd": JDRequirements(title="ML Engineer", required_skills=["Python"])}

    report = build_json_report(minimal_state)

    assert report["profiles"] == {}
    assert report["prefilter_results"] == {}


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


def test_build_markdown_report_orders_rows_by_rank_ascending():
    state = _sample_state()
    state["calibrated_results"] = [
        CalibratedResult(
            candidate_id="third", final_rank=3, tier=Tier.WEAK_FIT,
            rating=4, calibration_notes="Ranked third",
        ),
        CalibratedResult(
            candidate_id="first", final_rank=1, tier=Tier.STRONG_FIT,
            rating=9, calibration_notes="Ranked first",
        ),
        CalibratedResult(
            candidate_id="second", final_rank=2, tier=Tier.MODERATE_FIT,
            rating=6, calibration_notes="Ranked second",
        ),
    ]

    markdown = build_markdown_report(state)

    lines = markdown.splitlines()
    header, separator, *row_lines = lines
    candidate_order = [line.split("|")[2].strip() for line in row_lines]

    assert candidate_order == ["first", "second", "third"]
    assert lines.index("| 1 | first | Strong Fit | 9 | Ranked first |") < \
        lines.index("| 2 | second | Moderate Fit | 6 | Ranked second |") < \
        lines.index("| 3 | third | Weak Fit | 4 | Ranked third |")


def test_build_markdown_report_escapes_pipes_and_newlines_in_notes():
    state = _sample_state()
    state["calibrated_results"] = [
        CalibratedResult(
            candidate_id="strong", final_rank=1, tier=Tier.STRONG_FIT, rating=9,
            calibration_notes="Great fit | but watch out\nfor gaps in employment",
        )
    ]

    markdown = build_markdown_report(state)
    lines = markdown.splitlines()

    # Header + separator + exactly one data row: no extra rows from the embedded newline.
    assert len(lines) == 3
    data_row = lines[2]

    # No embedded newline leaked into the output.
    assert "\n" not in data_row

    # The literal pipe from the notes text was escaped (backslash-pipe), not left
    # as a bare separator that would split the notes into extra table columns.
    assert "Great fit \\| but watch out for gaps in employment" in data_row
    assert "Great fit | but" not in data_row

    # Exactly 5 well-formed columns: splitting on the escaped-pipe-protected row
    # (only unescaped pipes act as separators) yields the 5 data fields plus the
    # two empty strings from the leading/trailing pipe.
    unescaped_split = data_row.replace("\\|", "").split("|")
    assert len(unescaped_split) == 7  # "", rank, candidate, tier, rating, notes, ""
