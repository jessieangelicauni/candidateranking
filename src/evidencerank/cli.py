from pathlib import Path

import click
from dotenv import load_dotenv

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.graph import build_graph
from evidencerank.io import load_resume_text, load_text_file

from evaluation.report import write_json_report, write_markdown_report

load_dotenv()

OUT_JSON = "report.json"
OUT_MD = "report.md"
PREFILTER_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD = 85.0


def run_pipeline(jd_path, resumes_dir, llm_concurrency) -> dict:
    """Run the full pipeline once and return the final graph state.

    Shared by the `rank` command and `rank_stability`, which calls this
    repeatedly on the same JD/resumes to measure rank stability across runs.
    """
    jd_text = load_text_file(jd_path)
    jd_requirements = parse_jd(jd_text)

    raw_resumes = {
        pdf_path.stem: load_resume_text(pdf_path)
        for pdf_path in sorted(Path(resumes_dir).glob("*.pdf"))
    }

    graph = build_graph()
    return graph.invoke(
        {
            "jd": jd_requirements,
            "raw_resumes": raw_resumes,
            "prefilter_threshold": PREFILTER_THRESHOLD,
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
            "max_concurrency": llm_concurrency,
        }
    )


@click.command()
@click.option("--jd", "jd_path", required=True, type=click.Path(exists=True))
@click.option("--resumes-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--llm-concurrency", default=4, type=click.IntRange(min=1))
def rank(jd_path, resumes_dir, llm_concurrency):
    """Rank every resume in RESUMES_DIR against the job description at JD."""
    final_state = run_pipeline(jd_path, resumes_dir, llm_concurrency)

    write_json_report(final_state, OUT_JSON)
    write_markdown_report([OUT_JSON], OUT_MD)
    click.echo(f"Wrote {OUT_JSON} and {OUT_MD}")


@click.command()
@click.option(
    "--reports",
    "report_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
)
@click.option("--out", default="report.md", type=click.Path())
def report(report_paths, out):
    """Build the combined ranking + evaluation report from one or more report.json files.

    Pass --reports once per report.json path. One path gives rankings and
    pipeline stats only; repeat --reports for each additional run to also
    include rank stability across runs, e.g.:

        evidencerank-report --reports a.json --reports b.json --out report.md
    """
    write_markdown_report(list(report_paths), out)
    click.echo(f"Wrote {out}")


@click.command()
@click.option("--jd", "jd_path", required=True, type=click.Path(exists=True))
@click.option("--resumes-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--runs", default=3, type=click.IntRange(min=2))
@click.option("--llm-concurrency", default=4, type=click.IntRange(min=1))
@click.option("--out", default="report.md", type=click.Path())
def rank_stability(jd_path, resumes_dir, runs, llm_concurrency, out):
    """Run the pipeline RUNS times on the same JD/resumes and build a
    combined report that includes rank stability across the runs.

    Writes run1.json, run2.json, ... (one full report.json per run, never
    overwritten) alongside OUT, so each run stays available for inspection -
    not just the aggregated report.md.
    """
    report_paths = []
    for i in range(1, runs + 1):
        click.echo(f"Run {i}/{runs}...")
        final_state = run_pipeline(jd_path, resumes_dir, llm_concurrency)
        path = f"run{i}.json"
        write_json_report(final_state, path)
        report_paths.append(path)
        click.echo(f"Wrote {path}")

    write_markdown_report(report_paths, out)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    rank()
