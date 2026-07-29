# Eval Report CLI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--with-eval-report` flag to `evidencerank rank` so a single command can produce `report.json`/`report.md` and the Markdown evaluation report together.

**Architecture:** `src/evidencerank/cli.py`'s `rank` command gains two new `click` options (`--with-eval-report`, `--out-eval-report`). After writing `report.json`/`report.md` as it does today, if the flag is set, it locally imports and calls the existing `evaluation.report.write_eval_markdown_report` on the just-written `report.json` path. No changes to `evaluation/` or the pipeline graph/agents.

**Tech Stack:** Python 3.11, `click`, existing `evaluation.report.write_eval_markdown_report` (from the prior eval-metric-report feature).

## Global Constraints

- `--with-eval-report` defaults to off (opt-in) — plain `evidencerank rank` runs must be unaffected: no eval-report file written, no GEval metrics invoked, no `evaluation` import paid for.
- `--out-eval-report` defaults to `evaluation-metric.md`.
- The `evaluation.report` import must be local to the `if with_eval_report:` branch in `rank()`, not a top-of-file import — so running plain `rank` never imports `deepeval`/GEval machinery.
- On failure inside the eval-report step, let the exception propagate (command exits non-zero) — no try/except swallowing. `report.json`/`report.md` are already written by that point regardless.
- No changes to `evaluation/report.py`, `evaluation/cli.py`, `evaluation/metrics.py`, `evaluation/rank_stability.py`, `src/evidencerank/graph.py`, or any agent module.

---

### Task 1: `--with-eval-report` / `--out-eval-report` options on `rank`

**Files:**
- Modify: `src/evidencerank/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `evaluation.report.write_eval_markdown_report(report_paths: list[str | Path], path: str | Path) -> None` (existing, from `evaluation/report.py` — imported locally inside the new branch).
- Produces: `rank` command gains `with_eval_report: bool` and `out_eval_report: str` parameters; no other code depends on this task.

Current `src/evidencerank/cli.py` (for reference — you are modifying this exact file):

```python
from pathlib import Path

import click
from dotenv import load_dotenv

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.graph import build_graph
from evidencerank.io import load_resume_text, load_text_file
from evidencerank.report import write_json_report, write_markdown_report

load_dotenv()


@click.command()
@click.option("--jd", "jd_path", required=True, type=click.Path(exists=True))
@click.option("--resumes-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out-json", default="report.json", type=click.Path())
@click.option("--out-md", default="report.md", type=click.Path())
@click.option("--prefilter-threshold", default=0.5, type=float)
@click.option("--hallucination-threshold", default=85.0, type=float)
def rank(jd_path, resumes_dir, out_json, out_md, prefilter_threshold, hallucination_threshold):
    """Rank every resume in RESUMES_DIR against the job description at JD."""
    jd_text = load_text_file(jd_path)
    jd_requirements = parse_jd(jd_text)

    raw_resumes = {
        pdf_path.stem: load_resume_text(pdf_path)
        for pdf_path in sorted(Path(resumes_dir).glob("*.pdf"))
    }

    graph = build_graph()
    final_state = graph.invoke(
        {
            "jd": jd_requirements,
            "raw_resumes": raw_resumes,
            "prefilter_threshold": prefilter_threshold,
            "hallucination_threshold": hallucination_threshold,
        }
    )

    write_json_report(final_state, out_json)
    write_markdown_report(final_state, out_md)
    click.echo(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    rank()
```

Existing test fixture pattern in `tests/test_cli.py` (for reference — you are adding to this file): it mocks `evidencerank.cli.parse_jd` and `evidencerank.cli.build_graph` so no real LLM/graph execution happens, builds a fake `final_state` dict, and invokes `rank` via `click.testing.CliRunner`. Follow this exact pattern for the new tests — do not change the existing `test_rank_command_writes_json_and_markdown_reports` test.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (the file already has `_make_pdf`, `json`, `Path`, `MagicMock`, `CliRunner`, `rank`, `CalibratedResult`, `JDRequirements`, `Tier` imported/defined — reuse them):

```python
def test_rank_command_with_eval_report_flag_writes_eval_report(tmp_path, monkeypatch):
    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    _make_pdf(resumes_dir / "candidate1.pdf", "Candidate One\nPython, PyTorch")

    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_final_state = {
        "jd": fake_jd,
        "dropped": [],
        "judge_results": {},
        "calibrated_results": [
            CalibratedResult(
                candidate_id="candidate1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Only candidate",
            )
        ],
        "hallucination_reports": {},
    }
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = fake_final_state
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    out_eval_report = tmp_path / "evaluation-metric.md"
    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
            "--with-eval-report",
            "--out-eval-report", str(out_eval_report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_eval_report.exists()
    content = out_eval_report.read_text(encoding="utf-8")
    assert "## Pipeline Stats" in content
    assert "## GEval Metrics" in content


def test_rank_command_without_eval_report_flag_skips_eval_report(tmp_path, monkeypatch):
    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    _make_pdf(resumes_dir / "candidate1.pdf", "Candidate One\nPython, PyTorch")

    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_final_state = {
        "jd": fake_jd,
        "dropped": [],
        "judge_results": {},
        "calibrated_results": [
            CalibratedResult(
                candidate_id="candidate1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Only candidate",
            )
        ],
        "hallucination_reports": {},
    }
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = fake_final_state
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
        ],
    )

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "evaluation-metric.md").exists()
```

Note: `fake_final_state["judge_results"]` is empty in both tests (matching the existing test's fixture) — this means `compute_geval_scores` inside `write_eval_markdown_report` builds zero `LLMTestCase`s, so `metric.measure()` is never called for any of the three GEval metrics. No mocking of `evaluation.metrics` is needed in these tests; the first test exercises the real `write_eval_markdown_report` call end-to-end with zero LLM calls, and still validates that the file is written with real section headers.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k eval_report`
Expected: `test_rank_command_with_eval_report_flag_writes_eval_report` FAILs with `Error: No such option: --with-eval-report` (via `result.exit_code != 0` — check `result.output` in the assertion failure). `test_rank_command_without_eval_report_flag_skips_eval_report` should currently PASS already (since the flag doesn't exist, no eval-report file is ever written) — that's expected; it exists to lock in default-off behavior going forward, not to prove new behavior.

- [ ] **Step 3: Write minimal implementation**

Replace the full contents of `src/evidencerank/cli.py` with:

```python
from pathlib import Path

import click
from dotenv import load_dotenv

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.graph import build_graph
from evidencerank.io import load_resume_text, load_text_file
from evidencerank.report import write_json_report, write_markdown_report

load_dotenv()


@click.command()
@click.option("--jd", "jd_path", required=True, type=click.Path(exists=True))
@click.option("--resumes-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out-json", default="report.json", type=click.Path())
@click.option("--out-md", default="report.md", type=click.Path())
@click.option("--prefilter-threshold", default=0.5, type=float)
@click.option("--hallucination-threshold", default=85.0, type=float)
@click.option("--with-eval-report", is_flag=True, default=False)
@click.option("--out-eval-report", default="evaluation-metric.md", type=click.Path())
def rank(
    jd_path,
    resumes_dir,
    out_json,
    out_md,
    prefilter_threshold,
    hallucination_threshold,
    with_eval_report,
    out_eval_report,
):
    """Rank every resume in RESUMES_DIR against the job description at JD."""
    jd_text = load_text_file(jd_path)
    jd_requirements = parse_jd(jd_text)

    raw_resumes = {
        pdf_path.stem: load_resume_text(pdf_path)
        for pdf_path in sorted(Path(resumes_dir).glob("*.pdf"))
    }

    graph = build_graph()
    final_state = graph.invoke(
        {
            "jd": jd_requirements,
            "raw_resumes": raw_resumes,
            "prefilter_threshold": prefilter_threshold,
            "hallucination_threshold": hallucination_threshold,
        }
    )

    write_json_report(final_state, out_json)
    write_markdown_report(final_state, out_md)
    click.echo(f"Wrote {out_json} and {out_md}")

    if with_eval_report:
        from evaluation.report import write_eval_markdown_report

        write_eval_markdown_report([out_json], out_eval_report)
        click.echo(f"Wrote {out_eval_report}")


if __name__ == "__main__":
    rank()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (all 3 tests in the file: the pre-existing one plus both new ones)

- [ ] **Step 5: Update README**

In `README.md`, find the "Running the pipeline" section (it currently ends with the "Note:" paragraph about `report.json`'s `profiles` section, right before the "## Model configuration" heading). Add a new paragraph immediately after that "Note:" paragraph and before `## Model configuration`:

```markdown

Pass `--with-eval-report` to also generate the evaluation metric report
(`evaluation-metric.md` by default, override with `--out-eval-report`) in the same run — see
[Evaluation metric report](#evaluation-metric-report) below for what it contains. This
requires `ollama serve` running locally with the eval judge model available (same as the
standalone `evidencerank-eval-report` command).
```

In the existing "Evaluation metric report" section (added by the prior eval-metric-report feature), after the first code block (the single-run `evidencerank-eval-report --reports report.json --out evaluation-metric.md` example) and before the "Pass `--reports` once per report path" paragraph, add one sentence noting the new shortcut:

```markdown

If you're evaluating a single run right after producing it, `evidencerank rank
--with-eval-report` (see [Running the pipeline](#running-the-pipeline) above) does this in
one command instead of two.
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS (all existing tests plus the 2 new tests in `tests/test_cli.py`, no regressions)

- [ ] **Step 7: Commit**

```bash
git add src/evidencerank/cli.py tests/test_cli.py README.md
git commit -m "feat: add --with-eval-report flag to rank command"
```

---

## Manual Verification (after Task 1)

Run the combined command against the real resumes/JD in this repo, against a live Ollama server, to confirm the full path works end-to-end (not just mocked):

```bash
uv run evidencerank \
  --jd ai_data_engineer.txt \
  --resumes-dir resumes \
  --out-json /tmp/manual_report.json \
  --out-md /tmp/manual_report.md \
  --with-eval-report \
  --out-eval-report /tmp/manual_evaluation-metric.md
cat /tmp/manual_evaluation-metric.md
```

Expected: all three files are written; `manual_evaluation-metric.md` has `## Pipeline Stats` and `## GEval Metrics` tables populated with real numbers; no errors from the GEval Ollama calls; the command prints `Wrote /tmp/manual_report.json and /tmp/manual_report.md` followed by `Wrote /tmp/manual_evaluation-metric.md`.
