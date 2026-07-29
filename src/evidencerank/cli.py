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
PREFILTER_THRESHOLD = 0.9
HALLUCINATION_THRESHOLD = 85.0


def run_pipeline(jd_path, resumes_dir, llm_concurrency) -> dict:
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
            "raw_jd_text": jd_text,
            "raw_resumes": raw_resumes,
            "prefilter_threshold": PREFILTER_THRESHOLD,
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
            "max_concurrency": llm_concurrency,
        }
    )


@click.group()
def cli():
    pass


@cli.command()
@click.option("--jd", "jd_path", required=True, type=click.Path(exists=True))
@click.option("--resumes-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--llm-concurrency", default=12, type=click.IntRange(min=1))
@click.option("--runs", default=1, type=click.IntRange(min=1))
@click.option("--reports", "extra_report_paths", multiple=True, type=click.Path(exists=True))
@click.option("--out", default=OUT_MD, type=click.Path())
def rank(jd_path, resumes_dir, llm_concurrency, runs, extra_report_paths, out):
    """Rank every resume in RESUMES_DIR against the job description at JD.

    --runs 1 (default) writes report.json. --runs N>1 runs the pipeline N
    times and writes run1.json..runN.json instead (never overwritten). Pass
    --reports (repeatable) to also fold in existing report.json files - e.g.
    from past runs - into the rank stability comparison alongside this run's
    output. OUT gets rank stability (Spearman/Kendall-tau) whenever 2+
    reports (fresh + --reports combined) are being compared.
    """
    fresh_paths = []
    if runs == 1:
        final_state = run_pipeline(jd_path, resumes_dir, llm_concurrency)
        write_json_report(final_state, OUT_JSON)
        fresh_paths.append(OUT_JSON)
        click.echo(f"Wrote {OUT_JSON}")
    else:
        for i in range(1, runs + 1):
            click.echo(f"Run {i}/{runs}...")
            final_state = run_pipeline(jd_path, resumes_dir, llm_concurrency)
            path = f"run{i}.json"
            write_json_report(final_state, path)
            fresh_paths.append(path)
            click.echo(f"Wrote {path}")

    all_report_paths = fresh_paths + list(extra_report_paths)
    write_markdown_report(all_report_paths, out)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    cli()
