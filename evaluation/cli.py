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
@click.option("--out", default="eval_report.md", type=click.Path())
def eval_report(report_paths, out):
    """Build an evaluation metric report from one or more report.json files.

    Pass one path for GEval metrics + pipeline stats only, or two or more to
    also include rank stability across runs.
    """
    write_eval_markdown_report(list(report_paths), out)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    eval_report()
