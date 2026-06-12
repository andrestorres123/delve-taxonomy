"""Metadata exporter for run configuration and statistics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from delve.exporters.base import Exporter

if TYPE_CHECKING:
    from delve.result import DelveResult


class MetadataExporter(Exporter):
    """Export run metadata and configuration."""

    async def export(self, result: DelveResult, output_dir: str) -> Path:
        """Export metadata as JSON.

        Args:
            result: DelveResult to export
            output_dir: Directory for output file

        Returns:
            Path: Path to metadata.json file
        """
        dir_path = self.ensure_output_dir(output_dir)
        path = dir_path / "metadata.json"

        metadata = {
            "delve_version": "0.1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "configuration": {
                "model": result.config.model,
                "fast_llm": result.config.fast_llm,
                "sample_size": result.config.sample_size,
                "batch_size": result.config.batch_size,
                "use_case": result.config.use_case,
                "output_formats": result.config.output_formats,
                "output_dir": result.config.output_dir,
            },
            "statistics": result.metadata,
            "status_log": result.metadata.get("status_log", []),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return path
