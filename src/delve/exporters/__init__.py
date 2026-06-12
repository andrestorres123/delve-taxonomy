"""Result exporters for generating output files."""

from delve.exporters.base import Exporter
from delve.exporters.csv_exporter import CSVExporter
from delve.exporters.json_exporter import JSONExporter
from delve.exporters.markdown_exporter import MarkdownExporter
from delve.exporters.metadata_exporter import MetadataExporter

# Registry of available exporters
_EXPORTERS = {
    "json": JSONExporter(),
    "csv": CSVExporter(),
    "markdown": MarkdownExporter(),
    "metadata": MetadataExporter(),
}


def get_exporters():
    """Get registry of available exporters.

    Returns:
        dict: Mapping of format name to exporter instance
    """
    return _EXPORTERS


__all__ = [
    "Exporter",
    "JSONExporter",
    "CSVExporter",
    "MarkdownExporter",
    "MetadataExporter",
    "get_exporters",
]
