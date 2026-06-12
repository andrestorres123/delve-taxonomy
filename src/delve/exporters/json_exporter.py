"""JSON exporter for taxonomy and labeled documents."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from delve.exporters.base import Exporter

if TYPE_CHECKING:
    from delve.result import DelveResult


class JSONExporter(Exporter):
    """Export results as JSON files."""

    async def export(self, result: DelveResult, output_dir: str) -> Path:
        """Export taxonomy and labeled documents as JSON.

        Creates two files:
        - taxonomy.json: The generated taxonomy with metadata
        - labeled_documents.json: Documents with their assigned categories

        Args:
            result: DelveResult to export
            output_dir: Directory for output files

        Returns:
            Path: Path to the main taxonomy.json file
        """
        dir_path = self.ensure_output_dir(output_dir)

        # Export taxonomy
        taxonomy_path = dir_path / "taxonomy.json"
        taxonomy_data = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "metadata": result.metadata,
            "categories": [cat.to_dict() for cat in result.taxonomy],
        }
        with open(taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy_data, f, indent=2, ensure_ascii=False)

        # Export labeled documents
        docs_path = dir_path / "labeled_documents.json"
        docs_data = [
            {
                "id": doc.id,
                "category": doc.category,
                "summary": doc.summary,
                "content_preview": doc.content[:200] if doc.content else "",
            }
            for doc in result.labeled_documents
        ]
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)

        return taxonomy_path
