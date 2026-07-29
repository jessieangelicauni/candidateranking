import click

from evidencerank.cli import run_pipeline

from evaluation.report import write_json_report, write_markdown_report


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
    report()
