from pathlib import Path

import click
from dotenv import load_dotenv

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.graph import build_graph
from evidencerank.io import load_resume_text, load_text_file
from evidencerank.report import write_json_report, write_markdown_report

load_dotenv()

OUT_JSON = "report.json"
OUT_MD = "report.md"
OUT_EVAL_REPORT = "evaluation-metric.md"
PREFILTER_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD = 85.0


def run_pipeline(jd_path, resumes_dir, llm_concurrency) -> dict:
    """Run the full pipeline once and return the final graph state.

    Shared by the `rank` command and `evaluation.cli.rank_stability`, which
    calls this repeatedly on the same JD/resumes to measure rank stability
    across runs.
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
    write_markdown_report(final_state, OUT_MD)
    click.echo(f"Wrote {OUT_JSON} and {OUT_MD}")

    from evaluation.report import write_eval_markdown_report

    write_eval_markdown_report([OUT_JSON], OUT_EVAL_REPORT, max_concurrency=llm_concurrency)
    click.echo(f"Wrote {OUT_EVAL_REPORT}")


if __name__ == "__main__":
    rank()
