"""Command-line interface for Delve."""

import sys
from typing import Optional

import click

from delve import Delve, __version__
from delve.cli.utils import detect_source_type, print_summary, validate_file
from delve.console import Console, Verbosity
from delve.utils import validate_all_api_keys


@click.group()
@click.version_option(version=__version__)
def cli():
    r"""Delve - AI-powered taxonomy generation for your data.

    Generate taxonomies and automatically categorize your documents
    using state-of-the-art language models.

    \b
    Examples:
      # Basic CSV usage
      delve run data.csv --text-column conversation

      # JSON with JSONPath
      delve run data.json --json-path "$.messages[*].content"

      # LangSmith source
      delve run langsmith://my-project --langsmith-key $KEY --days 7
    """
    pass


@cli.command()
@click.argument("data_source", type=str)
@click.option(
    "--text-column",
    type=str,
    help="Column containing text data (for CSV/tabular)",
)
@click.option(
    "--id-column",
    type=str,
    help="Column containing document IDs (optional)",
)
@click.option(
    "--json-path",
    type=str,
    help="JSONPath expression for nested JSON (e.g., '$.messages[*].content')",
)
@click.option(
    "--source-type",
    type=click.Choice(["csv", "json", "jsonl", "langsmith", "auto"]),
    default="auto",
    help="Force specific data source type",
)
@click.option(
    "--model",
    default="anthropic/claude-opus-4-8",
    help="Main LLM model for reasoning",
)
@click.option(
    "--fast-llm",
    default="anthropic/claude-haiku-4-5",
    help="Fast LLM for summarization",
)
@click.option(
    "--sample-size",
    type=int,
    default=100,
    help="Number of documents to sample for taxonomy generation",
)
@click.option(
    "--batch-size",
    type=int,
    default=200,
    help="Batch size for processing",
)
@click.option(
    "--max-clusters",
    type=int,
    default=5,
    help="Maximum number of clusters/categories to generate",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./results",
    help="Output directory for results",
)
@click.option(
    "--output-format",
    multiple=True,
    type=click.Choice(["json", "csv", "markdown"]),
    default=["json", "csv", "markdown"],
    help="Output formats (can specify multiple)",
)
@click.option(
    "--use-case",
    type=str,
    help="Description of taxonomy use case",
)
@click.option(
    "--langsmith-key",
    type=str,
    envvar="LANGSMITH_API_KEY",
    help="LangSmith API key (for langsmith:// sources)",
)
@click.option(
    "--days",
    type=int,
    default=7,
    help="Days to look back (for LangSmith sources)",
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Only show errors",
)
@click.option(
    "-v", "--verbose",
    count=True,
    help="Increase verbosity (-v for progress bars, -vv for debug)",
)
def run(
    data_source: str,
    text_column: Optional[str],
    id_column: Optional[str],
    json_path: Optional[str],
    source_type: str,
    model: str,
    fast_llm: str,
    sample_size: int,
    batch_size: int,
    max_clusters: int,
    output_dir: str,
    output_format: tuple,
    use_case: Optional[str],
    langsmith_key: Optional[str],
    days: int,
    quiet: bool,
    verbose: int,
):
    r"""Run taxonomy generation on DATA_SOURCE.

    DATA_SOURCE can be:
      - Path to CSV file (e.g., data.csv)
      - Path to JSON/JSONL file (e.g., data.json)
      - LangSmith URI (e.g., langsmith://project-name)

    \b
    Examples:

      \b
      # Basic CSV usage
      delve run data.csv --text-column conversation

      \b
      # JSON with nested path
      delve run data.json --json-path "$.messages[*].content"

      \b
      # LangSmith source
      delve run langsmith://my-project --langsmith-key $KEY --days 7

      \b
      # Custom configuration
      delve run data.csv --text-column text --sample-size 200 \\
        --model anthropic/claude-opus-4-8 --output-dir ./output
    """
    # Determine verbosity level
    if quiet:
        verbosity = Verbosity.QUIET
    elif verbose >= 2:
        verbosity = Verbosity.DEBUG
    elif verbose == 1:
        verbosity = Verbosity.VERBOSE
    else:
        verbosity = Verbosity.NORMAL  # CLI default

    # Create console with appropriate verbosity
    console = Console(verbosity)

    # Auto-detect source type if needed
    if source_type == "auto":
        try:
            source_type = detect_source_type(data_source)
        except click.BadParameter as e:
            console.error(e.message)
            sys.exit(1)

    # Validate file-based sources
    if source_type in ("csv", "json", "jsonl"):
        try:
            validate_file(data_source)
        except click.BadParameter as e:
            console.error(e.message)
            sys.exit(1)

    # Validate required parameters
    if source_type == "csv" and not text_column:
        console.error("--text-column is required for CSV files")
        sys.exit(1)

    if source_type == "langsmith" and not langsmith_key:
        console.error(
            "--langsmith-key is required for LangSmith sources. "
            "Set LANGSMITH_API_KEY environment variable or use --langsmith-key option."
        )
        sys.exit(1)

    # Validate API keys early - before any processing starts
    # Required keys are derived from the model providers; embeddings (OpenAI)
    # are needed only when sample_size > 0 (classifier path).
    try:
        with console.status("Validating API keys..."):
            validate_all_api_keys([model, fast_llm], needs_embeddings=(sample_size > 0))
        console.success("API keys validated")
    except ValueError as e:
        console.error("Missing or invalid API keys")
        console.print()
        # Print each line of the error message
        for line in str(e).split("\n"):
            if line.strip():
                console.error(line)
        sys.exit(1)

    # Create Delve client with console
    delve_client = Delve(
        model=model,
        fast_llm=fast_llm,
        sample_size=sample_size,
        batch_size=batch_size,
        max_num_clusters=max_clusters,
        use_case=use_case,
        output_dir=output_dir,
        output_formats=list(output_format),
        verbosity=verbosity,
        console=console,
    )

    # Prepare adapter kwargs
    adapter_kwargs = {}
    if source_type == "langsmith":
        adapter_kwargs["api_key"] = langsmith_key
        adapter_kwargs["days"] = days
    elif source_type == "json" and json_path:
        adapter_kwargs["json_path"] = json_path

    # Run taxonomy generation
    try:
        # Show startup info in verbose mode
        console.info("Starting taxonomy generation...")
        console.info(f"  Source: {data_source}")
        console.info(f"  Model: {model}")
        console.info(f"  Sample size: {sample_size}")

        result = delve_client.run_sync(
            data_source,
            text_column=text_column,
            id_column=id_column,
            source_type=source_type,
            **adapter_kwargs,
        )

        # Print summary
        print_summary(result, output_dir, console)

    except ValueError as e:
        # User-friendly errors (like missing API key, validation errors)
        console.error(str(e))
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        error_lower = error_msg.lower()

        # Check for common error patterns and provide helpful messages
        if "api_key" in error_lower or "authentication" in error_lower or "auth_token" in error_lower:
            console.error("Authentication Error")
            console.error("The API key is missing or invalid.")
            console.error("Please set your Anthropic API key:")
            console.error("  export ANTHROPIC_API_KEY=your-api-key-here")
            console.error("You can get an API key from: https://console.anthropic.com/")
        elif "could not resolve" in error_lower and "authentication" in error_lower:
            console.error("API Key Not Found")
            console.error("The ANTHROPIC_API_KEY environment variable is not set.")
            console.error("Please set your API key:")
            console.error("  export ANTHROPIC_API_KEY=your-api-key-here")
            console.error("Get your API key: https://console.anthropic.com/")
        else:
            console.error(error_msg)

        # Show traceback in debug mode
        if verbosity >= Verbosity.DEBUG:
            import traceback
            console.debug("Full error details:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
