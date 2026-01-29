"""Result objects for taxonomy generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from pathlib import Path

import pandas as pd

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier
    from delve.state import Doc, State
    from delve.configuration import Configuration
else:
    from delve.state import State


@dataclass
class TaxonomyCategory:
    """A single taxonomy category."""

    id: str
    name: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }


@dataclass
class DelveResult:
    """Results from taxonomy generation."""

    taxonomy: List[TaxonomyCategory]
    labeled_documents: List[Doc]
    metadata: Dict[str, Any]
    config: Configuration
    export_paths: Dict[str, Path] = field(default_factory=dict)
    _classifier_model: Optional[Any] = field(default=None, repr=False)
    _classifier_index_to_category: Optional[Dict[int, str]] = field(default=None, repr=False)

    @classmethod
    def from_state(
        cls,
        state: Union[State, Dict[str, Any]],
        config: Configuration,
        run_duration: float = 0.0,
        source_info: Dict[str, Any] = None,
    ) -> DelveResult:
        """Create result from graph state.

        Args:
            state: The final state from graph execution (State dataclass or dict)
            config: Configuration used for the run
            run_duration: Total run time in seconds
            source_info: Information about the data source

        Returns:
            DelveResult instance
        """
        from collections import Counter

        # Handle both dict (from LangGraph) and State object
        if isinstance(state, dict):
            clusters = state.get("clusters", [])
            documents_raw = state.get("documents", [])
            status = state.get("status", [])
            classifier_metrics = state.get("classifier_metrics")
            llm_labeled_count = state.get("llm_labeled_count", 0)
            classifier_labeled_count = state.get("classifier_labeled_count", 0)
            skipped_document_count = state.get("skipped_document_count", 0)
            warnings = state.get("warnings", [])
            classifier_model = state.get("classifier_model")
            classifier_index_to_category = state.get("classifier_index_to_category")
        else:
            # State object
            clusters = state.clusters if state.clusters else []
            documents_raw = state.documents if state.documents else []
            status = state.status if state.status else []
            classifier_metrics = getattr(state, "classifier_metrics", None)
            llm_labeled_count = getattr(state, "llm_labeled_count", 0)
            classifier_labeled_count = getattr(state, "classifier_labeled_count", 0)
            skipped_document_count = getattr(state, "skipped_document_count", 0)
            warnings = getattr(state, "warnings", []) or []
            classifier_model = getattr(state, "classifier_model", None)
            classifier_index_to_category = getattr(state, "classifier_index_to_category", None)

        # Extract final taxonomy from clusters
        # clusters is a list of lists of dicts, get the last list (most recent taxonomy)
        final_clusters = []
        if clusters and len(clusters) > 0:
            final_clusters = clusters[-1]

        taxonomy = [
            TaxonomyCategory(
                id=str(c.get("id", "")),
                name=c.get("name", ""),
                description=c.get("description", ""),
            )
            for c in final_clusters
            if isinstance(c, dict)
        ]

        # Convert documents to Doc objects if they're dicts
        from delve.state import Doc
        documents = []
        for doc in documents_raw:
            if isinstance(doc, dict):
                documents.append(Doc(
                    id=doc.get("id", ""),
                    content=doc.get("content", ""),
                    summary=doc.get("summary"),
                    explanation=doc.get("explanation"),
                    category=doc.get("category"),
                ))
            else:
                # Already a Doc object
                documents.append(doc)

        # Calculate category distribution
        category_counts = dict(Counter(
            doc.category for doc in documents if doc.category
        ))

        # Build enhanced metadata
        metadata: Dict[str, Any] = {
            # Basic info
            "num_documents": len(documents),
            "num_categories": len(taxonomy),
            "sample_size": config.sample_size,
            "batch_size": config.batch_size,
            "model": config.model,
            "fast_llm": config.fast_llm,
            "status_log": status if status else [],

            # Timing
            "run_duration_seconds": round(run_duration, 2),

            # Category distribution
            "category_counts": category_counts,

            # Labeling breakdown
            "llm_labeled_count": llm_labeled_count,
            "classifier_labeled_count": classifier_labeled_count,
            "skipped_document_count": skipped_document_count,

            # Quality info
            "warnings": warnings,
        }

        # Add classifier metrics if available
        if classifier_metrics:
            metadata["classifier_metrics"] = {
                "train_accuracy": round(classifier_metrics.get("train_accuracy", 0), 4),
                "test_accuracy": round(classifier_metrics.get("test_accuracy", 0), 4),
                "train_f1": round(classifier_metrics.get("train_f1", 0), 4),
                "test_f1": round(classifier_metrics.get("test_f1", 0), 4),
            }

        # Add source info if provided
        if source_info:
            metadata["source"] = source_info

        result = cls(
            taxonomy=taxonomy,
            labeled_documents=documents,
            metadata=metadata,
            config=config,
        )
        # Store classifier for potential export (not in constructor to keep clean API)
        result._classifier_model = classifier_model
        result._classifier_index_to_category = classifier_index_to_category
        return result

    async def export(self) -> Dict[str, Path]:
        """Export results in configured formats.

        Returns:
            Dict mapping format name to output file path
        """
        from delve.exporters import get_exporters

        output_paths = {}
        exporters = get_exporters()

        for format_name in self.config.output_formats:
            if format_name in exporters:
                exporter = exporters[format_name]
                path = await exporter.export(self, self.config.output_dir)
                output_paths[format_name] = path

        # Always export metadata
        if "metadata" in exporters:
            metadata_path = await exporters["metadata"].export(self, self.config.output_dir)
            output_paths["metadata"] = metadata_path

        self.export_paths = output_paths
        return output_paths

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "taxonomy": [cat.to_dict() for cat in self.taxonomy],
            "labeled_documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "category": doc.category,
                    "summary": doc.summary,
                    "explanation": doc.explanation,
                }
                for doc in self.labeled_documents
            ],
            "metadata": self.metadata,
        }

    def save_classifier(self, path: Union[str, Path]) -> Path:
        """Save trained classifier to joblib file for later reuse.

        The saved bundle includes the trained model, category mappings,
        embedding configuration, and taxonomy. Use Delve.classify() to
        load and apply it to new documents.

        Args:
            path: Output file path (should end with .joblib)

        Returns:
            Path to saved classifier bundle

        Raises:
            ValueError: If no classifier is available (all docs were LLM-labeled)

        Example:
            >>> result = delve.run_sync("data.csv", text_column="text")
            >>> result.save_classifier("my_classifier.joblib")
        """
        from delve.core.classifier import ClassifierBundle, save_bundle
        import delve

        if self._classifier_model is None:
            raise ValueError(
                "No classifier available to save. This happens when all documents "
                "were labeled by the LLM (sample_size >= total documents). "
                "A classifier is only trained when sample_size < total documents."
            )

        # Determine embedding dimensions from model config
        # text-embedding-3-large defaults to 3072
        embedding_dims = 3072
        if "small" in self.config.embedding_model:
            embedding_dims = 1536

        bundle = ClassifierBundle(
            model=self._classifier_model,
            index_to_category=self._classifier_index_to_category,
            embedding_model=self.config.embedding_model,
            embedding_dimensions=embedding_dims,
            taxonomy=[cat.to_dict() for cat in self.taxonomy],
            metrics=self.metadata.get("classifier_metrics", {}),
            created_at=datetime.now().isoformat(),
            delve_version=delve.__version__,
        )
        return save_bundle(bundle, path)


@dataclass
class ClassificationResult:
    """Results from classifying documents with a saved classifier."""

    documents: List[Doc]
    classifier_info: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns:
            DataFrame with columns: id, content, category, confidence
        """
        return pd.DataFrame([
            {
                "id": doc.id,
                "content": doc.content,
                "category": doc.category,
                "confidence": doc.confidence,
            }
            for doc in self.documents
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "category": doc.category,
                    "confidence": doc.confidence,
                }
                for doc in self.documents
            ],
            "classifier_info": self.classifier_info,
        }

    def export(
        self,
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """Export classification results to files.

        Args:
            output_dir: Directory for output files
            formats: List of formats (default: ["csv"])
                Supported: "csv", "json"

        Returns:
            Dict mapping format name to output file path
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        formats = formats or ["csv"]
        output_paths = {}

        if "csv" in formats:
            csv_path = output_dir / "classified_documents.csv"
            self.to_dataframe().to_csv(csv_path, index=False)
            output_paths["csv"] = csv_path

        if "json" in formats:
            json_path = output_dir / "classified_documents.json"
            with open(json_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            output_paths["json"] = json_path

        return output_paths


@dataclass
class TrainingResult:
    """Results from training a classifier from labeled data."""

    model: Any  # RandomForestClassifier
    index_to_category: Dict[int, str]
    taxonomy: List[TaxonomyCategory]
    metrics: Dict[str, Any]
    training_docs_count: int
    validation_docs_count: int
    embedding_model: str
    created_at: str

    def save_classifier(self, path: Union[str, Path]) -> Path:
        """Save trained classifier to joblib file.

        Args:
            path: Output file path (should end with .joblib)

        Returns:
            Path to saved classifier bundle
        """
        from delve.core.classifier import ClassifierBundle, save_bundle
        import delve

        # Determine embedding dimensions
        embedding_dims = 3072
        if "small" in self.embedding_model:
            embedding_dims = 1536

        bundle = ClassifierBundle(
            model=self.model,
            index_to_category=self.index_to_category,
            embedding_model=self.embedding_model,
            embedding_dimensions=embedding_dims,
            taxonomy=[cat.to_dict() for cat in self.taxonomy],
            metrics=self.metrics,
            created_at=self.created_at,
            delve_version=delve.__version__,
        )
        return save_bundle(bundle, path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (excludes model)."""
        return {
            "taxonomy": [cat.to_dict() for cat in self.taxonomy],
            "metrics": self.metrics,
            "training_docs_count": self.training_docs_count,
            "validation_docs_count": self.validation_docs_count,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
        }
