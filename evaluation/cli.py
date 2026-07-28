import click

from evaluation.report import write_eval_markdown_report


@click.command()
@click.option(
    "--reports",
    "report_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
)
@click.option("--out", default="evaluation-metric.md", type=click.Path())
def eval_report(report_paths, out):
    """Build an evaluation metric report from one or more report.json files.

    Pass --reports once per report.json path. One path gives GEval metrics +
    pipeline stats only; repeat --reports for each additional run to also
    include rank stability across runs, e.g.:

        evidencerank-eval-report --reports a.json --reports b.json --out evaluation-metric.md
    """
    write_eval_markdown_report(list(report_paths), out)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    eval_report()
