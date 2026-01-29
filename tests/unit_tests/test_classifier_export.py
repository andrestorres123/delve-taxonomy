"""Tests for classifier export/load functionality."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from delve.core.classifier import (
    ClassifierBundle,
    save_bundle,
    load_bundle,
    _infer_taxonomy_from_labels,
)
from delve.state import Doc
from delve.result import ClassificationResult, TrainingResult, TaxonomyCategory, DelveResult
from delve.configuration import Configuration


class TestClassifierBundle:
    """Test ClassifierBundle dataclass and save/load operations."""

    @pytest.fixture
    def sample_bundle(self):
        """Create a sample classifier bundle for testing."""
        # Create a minimal trained RandomForest
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1])
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        return ClassifierBundle(
            model=model,
            index_to_category={0: "Bug", 1: "Feature", 2: "Documentation"},
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
            taxonomy=[
                {"id": "1", "name": "Bug", "description": "Software defects"},
                {"id": "2", "name": "Feature", "description": "New functionality"},
                {"id": "3", "name": "Documentation", "description": "Documentation updates"},
            ],
            metrics={"test_accuracy": 0.85, "test_f1": 0.83},
            created_at="2025-01-29T12:00:00",
            delve_version="0.1.11",
        )

    def test_classifier_bundle_round_trip(self, sample_bundle):
        """Test that saving and loading a bundle preserves all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "classifier.joblib"

            # Save
            saved_path = save_bundle(sample_bundle, path)
            assert saved_path == path
            assert path.exists()

            # Load
            loaded_bundle = load_bundle(path)

            # Verify all fields match
            assert loaded_bundle.index_to_category == sample_bundle.index_to_category
            assert loaded_bundle.embedding_model == sample_bundle.embedding_model
            assert loaded_bundle.embedding_dimensions == sample_bundle.embedding_dimensions
            assert loaded_bundle.taxonomy == sample_bundle.taxonomy
            assert loaded_bundle.metrics == sample_bundle.metrics
            assert loaded_bundle.created_at == sample_bundle.created_at
            assert loaded_bundle.delve_version == sample_bundle.delve_version

            # Verify model works
            X_test = np.random.randn(5, 10)
            original_preds = sample_bundle.model.predict(X_test)
            loaded_preds = loaded_bundle.model.predict(X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_save_bundle_creates_directories(self, sample_bundle):
        """Test that save_bundle creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "classifier.joblib"
            saved_path = save_bundle(sample_bundle, path)
            assert saved_path == path
            assert path.exists()

    def test_load_bundle_file_not_found(self):
        """Test that load_bundle raises error for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_bundle("/nonexistent/path/classifier.joblib")

    def test_load_bundle_invalid_format(self):
        """Test that load_bundle raises error for invalid format."""
        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.joblib"
            joblib.dump({"not": "a bundle"}, path)

            with pytest.raises(ValueError, match="Invalid classifier bundle format"):
                load_bundle(path)


class TestInferTaxonomy:
    """Test taxonomy inference from labels."""

    def test_infer_taxonomy_from_labels_basic(self):
        """Test basic taxonomy inference."""
        labels = ["Bug", "Feature", "Bug", "Documentation", "Feature"]
        taxonomy = _infer_taxonomy_from_labels(labels)

        assert len(taxonomy) == 3
        names = {cat["name"] for cat in taxonomy}
        assert names == {"Bug", "Documentation", "Feature"}

        # Check structure
        for cat in taxonomy:
            assert "id" in cat
            assert "name" in cat
            assert "description" in cat

    def test_infer_taxonomy_preserves_order(self):
        """Test that inferred taxonomy is sorted alphabetically."""
        labels = ["Zebra", "Apple", "Mango"]
        taxonomy = _infer_taxonomy_from_labels(labels)

        names = [cat["name"] for cat in taxonomy]
        assert names == ["Apple", "Mango", "Zebra"]

    def test_infer_taxonomy_single_label(self):
        """Test inference with single unique label."""
        labels = ["Bug", "Bug", "Bug"]
        taxonomy = _infer_taxonomy_from_labels(labels)

        assert len(taxonomy) == 1
        assert taxonomy[0]["name"] == "Bug"


class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample classification result."""
        docs = [
            Doc(id="1", content="Fix crash", category="Bug", confidence=0.95),
            Doc(id="2", content="Add feature", category="Feature", confidence=0.88),
            Doc(id="3", content="Update docs", category="Documentation", confidence=0.72),
        ]
        return ClassificationResult(
            documents=docs,
            classifier_info={
                "classifier_path": "test.joblib",
                "embedding_model": "text-embedding-3-large",
                "num_categories": 3,
            },
        )

    def test_to_dataframe(self, sample_result):
        """Test conversion to DataFrame."""
        df = sample_result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["id", "content", "category", "confidence"]
        assert df["id"].tolist() == ["1", "2", "3"]
        assert df["confidence"].tolist() == [0.95, 0.88, 0.72]

    def test_to_dict(self, sample_result):
        """Test conversion to dictionary."""
        d = sample_result.to_dict()

        assert "documents" in d
        assert "classifier_info" in d
        assert len(d["documents"]) == 3

    def test_export_csv(self, sample_result):
        """Test exporting to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = sample_result.export(tmpdir, formats=["csv"])

            assert "csv" in paths
            assert paths["csv"].exists()

            # Verify content
            df = pd.read_csv(paths["csv"])
            assert len(df) == 3

    def test_export_json(self, sample_result):
        """Test exporting to JSON."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = sample_result.export(tmpdir, formats=["json"])

            assert "json" in paths
            assert paths["json"].exists()

            with open(paths["json"]) as f:
                data = json.load(f)
            assert len(data["documents"]) == 3


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    @pytest.fixture
    def sample_training_result(self):
        """Create sample training result."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y = np.array([0] * 10 + [1] * 10)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        return TrainingResult(
            model=model,
            index_to_category={0: "Bug", 1: "Feature"},
            taxonomy=[
                TaxonomyCategory(id="1", name="Bug", description="Software defects"),
                TaxonomyCategory(id="2", name="Feature", description="New functionality"),
            ],
            metrics={"test_accuracy": 0.9, "test_f1": 0.88, "per_class_f1": {"Bug": 0.9, "Feature": 0.86}},
            training_docs_count=16,
            validation_docs_count=4,
            embedding_model="text-embedding-3-large",
            created_at="2025-01-29T12:00:00",
        )

    def test_save_classifier(self, sample_training_result):
        """Test saving classifier from training result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trained.joblib"
            saved_path = sample_training_result.save_classifier(path)

            assert saved_path == path
            assert path.exists()

            # Verify can be loaded
            bundle = load_bundle(path)
            assert bundle.embedding_model == "text-embedding-3-large"
            assert len(bundle.taxonomy) == 2

    def test_to_dict(self, sample_training_result):
        """Test conversion to dictionary."""
        d = sample_training_result.to_dict()

        assert "taxonomy" in d
        assert "metrics" in d
        assert "training_docs_count" in d
        assert d["training_docs_count"] == 16
        assert d["validation_docs_count"] == 4


class TestDelveResultSaveClassifier:
    """Test DelveResult.save_classifier() method."""

    @pytest.fixture
    def sample_delve_result(self):
        """Create sample Delve result with classifier."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y = np.array([0] * 10 + [1] * 10)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = DelveResult(
            taxonomy=[
                TaxonomyCategory(id="1", name="Bug", description="Software defects"),
                TaxonomyCategory(id="2", name="Feature", description="New functionality"),
            ],
            labeled_documents=[
                Doc(id="1", content="Fix crash", category="Bug"),
                Doc(id="2", content="Add button", category="Feature"),
            ],
            metadata={
                "num_documents": 100,
                "classifier_metrics": {"test_accuracy": 0.9, "test_f1": 0.88},
            },
            config=Configuration(sample_size=50),
        )
        result._classifier_model = model
        result._classifier_index_to_category = {0: "Bug", 1: "Feature"}
        return result

    def test_save_classifier_success(self, sample_delve_result):
        """Test successful classifier save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "classifier.joblib"
            saved_path = sample_delve_result.save_classifier(path)

            assert saved_path == path
            assert path.exists()

            # Verify bundle contents
            bundle = load_bundle(path)
            assert len(bundle.taxonomy) == 2
            assert bundle.embedding_model == "text-embedding-3-large"

    def test_save_classifier_no_classifier_error(self):
        """Test error when no classifier is available."""
        result = DelveResult(
            taxonomy=[TaxonomyCategory(id="1", name="Bug", description="Bugs")],
            labeled_documents=[Doc(id="1", content="Content", category="Bug")],
            metadata={"num_documents": 10},
            config=Configuration(sample_size=100),
        )
        # No classifier was trained (all docs LLM-labeled)

        with pytest.raises(ValueError, match="No classifier available"):
            result.save_classifier("test.joblib")


class TestDelveClassifyMethod:
    """Test Delve.classify() class method."""

    @pytest.fixture
    def sample_bundle_path(self):
        """Create a saved classifier bundle for testing."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y = np.array([0] * 7 + [1] * 7 + [2] * 6)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        bundle = ClassifierBundle(
            model=model,
            index_to_category={0: "Bug", 1: "Feature", 2: "Documentation"},
            embedding_model="text-embedding-3-small",  # Use small for faster tests
            embedding_dimensions=1536,
            taxonomy=[
                {"id": "1", "name": "Bug", "description": "Software defects"},
                {"id": "2", "name": "Feature", "description": "New functionality"},
                {"id": "3", "name": "Documentation", "description": "Doc updates"},
            ],
            metrics={"test_accuracy": 0.85},
            created_at="2025-01-29T12:00:00",
            delve_version="0.1.11",
        )

        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "classifier.joblib"
        save_bundle(bundle, path)
        return path

    @pytest.mark.asyncio
    async def test_classify_async_with_docs(self, sample_bundle_path):
        """Test classify_async with Doc objects (mocked embeddings)."""
        from delve.client import Delve

        docs = [
            Doc(id="1", content="Fix crash on startup"),
            Doc(id="2", content="Add dark mode feature"),
        ]

        # Mock the embeddings generation - patch where it's imported
        with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
            mock_encoder = Mock()
            mock_encoder.aembed_documents = AsyncMock(
                return_value=[np.random.randn(10).tolist() for _ in docs]
            )
            mock_embeddings.return_value = mock_encoder

            result = await Delve.classify_async(
                docs,
                classifier_path=sample_bundle_path,
            )

        assert isinstance(result, ClassificationResult)
        assert len(result.documents) == 2
        assert all(doc.category is not None for doc in result.documents)
        assert all(doc.confidence is not None for doc in result.documents)

    @pytest.mark.asyncio
    async def test_classify_async_with_dataframe(self, sample_bundle_path):
        """Test classify_async with DataFrame input (mocked embeddings)."""
        from delve.client import Delve

        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "text": ["Fix bug", "Add feature", "Update docs"],
        })

        with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
            mock_encoder = Mock()
            mock_encoder.aembed_documents = AsyncMock(
                return_value=[np.random.randn(10).tolist() for _ in range(3)]
            )
            mock_embeddings.return_value = mock_encoder

            result = await Delve.classify_async(
                df,
                classifier_path=sample_bundle_path,
                text_column="text",
                id_column="id",
            )

        assert len(result.documents) == 3
        assert result.documents[0].id == "1"

    def test_classify_sync(self, sample_bundle_path):
        """Test sync classify method (mocked embeddings)."""
        from delve.client import Delve

        docs = [Doc(id="1", content="Test content")]

        with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
            mock_encoder = Mock()
            mock_encoder.aembed_documents = AsyncMock(
                return_value=[np.random.randn(10).tolist()]
            )
            mock_embeddings.return_value = mock_encoder

            result = Delve.classify(
                docs,
                classifier_path=sample_bundle_path,
            )

        assert isinstance(result, ClassificationResult)
        assert len(result.documents) == 1


class TestDelveTrainFromLabeled:
    """Test Delve.train_from_labeled() class method."""

    @pytest.fixture
    def sample_labeled_csv(self):
        """Create a sample labeled CSV file."""
        df = pd.DataFrame({
            "id": [str(i) for i in range(20)],
            "text": [f"Document content {i}" for i in range(20)],
            "category": ["Bug"] * 7 + ["Feature"] * 7 + ["Documentation"] * 6,
        })
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "labeled.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.mark.asyncio
    async def test_train_from_labeled_async_basic(self, sample_labeled_csv):
        """Test basic training from labeled data (mocked embeddings)."""
        from delve.client import Delve

        with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
            mock_encoder = Mock()
            # Return embeddings that cluster by category
            embeddings = []
            for i in range(20):
                if i < 7:  # Bug
                    embeddings.append([1.0, 0.0, 0.0] + [0.0] * 7 + [np.random.randn() * 0.1])
                elif i < 14:  # Feature
                    embeddings.append([0.0, 1.0, 0.0] + [0.0] * 7 + [np.random.randn() * 0.1])
                else:  # Documentation
                    embeddings.append([0.0, 0.0, 1.0] + [0.0] * 7 + [np.random.randn() * 0.1])
            mock_encoder.aembed_documents = AsyncMock(return_value=embeddings)
            mock_embeddings.return_value = mock_encoder

            result = await Delve.train_from_labeled_async(
                sample_labeled_csv,
                text_column="text",
                label_column="category",
                id_column="id",
            )

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert len(result.taxonomy) == 3
        assert "test_accuracy" in result.metrics
        assert "test_f1" in result.metrics

    @pytest.mark.asyncio
    async def test_train_from_labeled_with_explicit_taxonomy(self, sample_labeled_csv):
        """Test training with explicit taxonomy provided."""
        from delve.client import Delve

        taxonomy = [
            {"id": "1", "name": "Bug", "description": "Software bugs and defects"},
            {"id": "2", "name": "Feature", "description": "New features and enhancements"},
            {"id": "3", "name": "Documentation", "description": "Documentation changes"},
        ]

        with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
            mock_encoder = Mock()
            mock_encoder.aembed_documents = AsyncMock(
                return_value=[np.random.randn(10).tolist() for _ in range(20)]
            )
            mock_embeddings.return_value = mock_encoder

            result = await Delve.train_from_labeled_async(
                sample_labeled_csv,
                text_column="text",
                label_column="category",
                taxonomy=taxonomy,
            )

        # Check that taxonomy descriptions are preserved
        tax_descs = {cat.name: cat.description for cat in result.taxonomy}
        assert tax_descs["Bug"] == "Software bugs and defects"

    @pytest.mark.asyncio
    async def test_train_from_labeled_invalid_column(self, sample_labeled_csv):
        """Test error handling for invalid column names."""
        from delve.client import Delve

        with pytest.raises(ValueError, match="Text column 'nonexistent' not found"):
            await Delve.train_from_labeled_async(
                sample_labeled_csv,
                text_column="nonexistent",
                label_column="category",
            )

    @pytest.mark.asyncio
    async def test_train_from_labeled_label_not_in_taxonomy(self, sample_labeled_csv):
        """Test error when labels don't match provided taxonomy."""
        from delve.client import Delve

        taxonomy = [
            {"id": "1", "name": "Bug", "description": "Bugs"},
            # Missing Feature and Documentation
        ]

        with pytest.raises(ValueError, match="Labels not in taxonomy"):
            await Delve.train_from_labeled_async(
                sample_labeled_csv,
                text_column="text",
                label_column="category",
                taxonomy=taxonomy,
            )

    def test_train_from_labeled_sync(self, sample_labeled_csv):
        """Test sync version of train_from_labeled."""
        from delve.client import Delve

        with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
            mock_encoder = Mock()
            mock_encoder.aembed_documents = AsyncMock(
                return_value=[np.random.randn(10).tolist() for _ in range(20)]
            )
            mock_embeddings.return_value = mock_encoder

            result = Delve.train_from_labeled(
                sample_labeled_csv,
                text_column="text",
                label_column="category",
            )

        assert isinstance(result, TrainingResult)


class TestLoadDocsForClassification:
    """Test helper method for loading docs."""

    def test_load_from_doc_list(self):
        """Test loading from list of Doc objects."""
        from delve.client import Delve

        docs = [Doc(id="1", content="Test")]
        result = Delve._load_docs_for_classification(docs, None, None)
        assert result == docs

    def test_load_from_dataframe(self):
        """Test loading from DataFrame."""
        from delve.client import Delve

        df = pd.DataFrame({
            "my_id": ["1", "2"],
            "my_text": ["Content 1", "Content 2"],
        })

        docs = Delve._load_docs_for_classification(
            df, text_column="my_text", id_column="my_id"
        )

        assert len(docs) == 2
        assert docs[0].id == "1"
        assert docs[0].content == "Content 1"

    def test_load_missing_text_column_error(self):
        """Test error when text_column not provided for DataFrame."""
        from delve.client import Delve

        df = pd.DataFrame({"text": ["Content"]})

        with pytest.raises(ValueError, match="text_column is required"):
            Delve._load_docs_for_classification(df, None, None)

    def test_load_from_csv(self):
        """Test loading from CSV file."""
        from delve.client import Delve

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            pd.DataFrame({
                "text": ["Content 1", "Content 2"],
            }).to_csv(path, index=False)

            docs = Delve._load_docs_for_classification(
                path, text_column="text", id_column=None
            )

            assert len(docs) == 2
