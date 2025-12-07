"""
Unified CLI entry point.

Usage:
    python -m src.cli train single ...
    python -m src.cli evaluate model ...

Future: Can configure entry point in pyproject.toml:
    [project.scripts]
    frost-cli = "src.cli:main"
"""

import click

# Import command groups (will be implemented in Phase 2)
try:
    from src.cli.commands.train import train
except ImportError:
    train = None

try:
    from src.cli.commands.evaluate import evaluate
except ImportError:
    evaluate = None

try:
    from src.cli.commands.analysis import analysis
except ImportError:
    analysis = None

try:
    from src.cli.commands.tools import tools
except ImportError:
    tools = None

try:
    from src.cli.commands.inference import inference
except ImportError:
    inference = None


@click.group()
def cli():
    """Frost risk forecasting CLI root."""
    pass


# Register command groups (if implemented)
if train is not None:
    cli.add_command(train, name="train")

if evaluate is not None:
    cli.add_command(evaluate, name="evaluate")

if analysis is not None:
    cli.add_command(analysis, name="analysis")

if tools is not None:
    cli.add_command(tools, name="tools")

if inference is not None:
    cli.add_command(inference, name="inference")


def main():
    """CLI entry function."""
    cli()


if __name__ == "__main__":
    main()

