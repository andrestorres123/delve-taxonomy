"""Markdown exporter for human-readable reports."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from delve.exporters.base import Exporter

if TYPE_CHECKING:
    from delve.result import DelveResult


class MarkdownExporter(Exporter):
    """Export results as human-readable Markdown report."""

    async def export(self, result: DelveResult, output_dir: str) -> Path:
        """Generate Markdown report.

        Args:
            result: DelveResult to export
            output_dir: Directory for output file

        Returns:
            Path: Path to the report.md file
        """
        dir_path = self.ensure_output_dir(output_dir)
        path = dir_path / "report.md"

        # Calculate statistics
        category_counts = Counter(doc.category for doc in result.labeled_documents)

        # Generate report
        report = self._generate_report(result, category_counts)

        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

        return path

    def _generate_report(self, result: DelveResult, category_counts: Counter) -> str:
        """Generate formatted report.

        Args:
            result: DelveResult to format
            category_counts: Category distribution

        Returns:
            str: Formatted markdown report
        """
        lines = [
            "# Taxonomy Generation Report",
            "",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Model:** {result.metadata['model']}",
            f"**Documents Processed:** {result.metadata['num_documents']}",
            "",
            "## Taxonomy Overview",
            "",
            f"Generated **{len(result.taxonomy)} categories** from analysis of {result.metadata['sample_size']} sampled documents.",
            "",
            "## Categories",
            "",
        ]

        # List each category with statistics
        for cat in result.taxonomy:
            count = category_counts.get(cat.name, 0)
            percentage = (count / len(result.labeled_documents) * 100) if result.labeled_documents else 0

            lines.extend([
                f"### {cat.name}",
                "",
                f"**Description:** {cat.description}",
                "",
                f"**Documents:** {count} ({percentage:.1f}%)",
                "",
            ])

        # Distribution chart
        lines.extend([
            "## Category Distribution",
            "",
            "| Category | Count | Percentage |",
            "|----------|-------|------------|",
        ])

        for cat in result.taxonomy:
            count = category_counts.get(cat.name, 0)
            percentage = (count / len(result.labeled_documents) * 100) if result.labeled_documents else 0
            lines.append(f"| {cat.name} | {count} | {percentage:.1f}% |")

        lines.extend([
            "",
            "## Sample Documents by Category",
            "",
        ])

        # Show 2 examples per category
        for cat in result.taxonomy:
            matching_docs = [
                doc for doc in result.labeled_documents
                if doc.category == cat.name
            ][:2]

            if matching_docs:
                lines.append(f"### {cat.name}")
                lines.append("")
                for doc in matching_docs:
                    preview = doc.content[:150].replace("\n", " ")
                    lines.append(f"- **{doc.id}**: {preview}...")
                    if doc.summary:
                        lines.append(f"  - *Summary:* {doc.summary}")
                lines.append("")

        # Configuration section
        lines.extend([
            "## Configuration",
            "",
            f"- **Sample Size:** {result.metadata['sample_size']}",
            f"- **Batch Size:** {result.metadata['batch_size']}",
            f"- **Model:** {result.metadata['model']}",
            "",
        ])

        return "\n".join(lines)
