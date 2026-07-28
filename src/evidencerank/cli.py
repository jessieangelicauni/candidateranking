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
@click.option("--llm-concurrency", default=4, type=int)
@click.option("--with-eval-report", is_flag=True, default=False)
@click.option("--out-eval-report", default="eval_report.md", type=click.Path())
def rank(
    jd_path,
    resumes_dir,
    out_json,
    out_md,
    prefilter_threshold,
    hallucination_threshold,
    llm_concurrency,
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
            "max_concurrency": llm_concurrency,
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
