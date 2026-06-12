"""Node for saving results using exporters."""

from langchain_core.runnables import RunnableConfig

from delve.configuration import Configuration
from delve.exporters import get_exporters
from delve.result import DelveResult
from delve.state import State


async def save_results(state: State, config: RunnableConfig) -> dict:
    """Save results using configured exporters.

    This is the final node in the graph that exports results
    in all configured formats.

    Args:
        state: Current application state with final results
        config: Configuration for the run

    Returns:
        dict: Updated state with export status
    """
    configuration = Configuration.from_runnable_config(config)

    # Create result object from state
    result = DelveResult.from_state(state, configuration)

    # Export in all configured formats
    exporters = get_exporters()
    export_paths = {}

    for format_name in configuration.output_formats:
        if format_name in exporters:
            exporter = exporters[format_name]
            path = await exporter.export(result, configuration.output_dir)
            export_paths[format_name] = str(path)

    # Always export metadata
    if "metadata" in exporters:
        metadata_path = await exporters["metadata"].export(result, configuration.output_dir)
        export_paths["metadata"] = str(metadata_path)

    status_message = f"Results saved to {configuration.output_dir}/ in formats: {', '.join(export_paths.keys())}"

    return {
        "status": [status_message],
    }
