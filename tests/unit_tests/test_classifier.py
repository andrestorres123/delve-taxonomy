"""Tests for classifier module."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from delve.core.classifier import (
    train_classifier,
    predict_with_classifier,
    get_prediction_confidence,
)
from delve.state import Doc


class TestClassifier:
    """Test the classifier training and prediction functions."""

    @pytest.fixture
    def sample_taxonomy(self):
        """Create sample taxonomy for testing."""
        return [
            {"id": "1", "name": "Bug", "description": "Software defects"},
            {"id": "2", "name": "Feature", "description": "New functionality"},
            {"id": "3", "name": "Documentation", "description": "Documentation updates"},
        ]

    @pytest.fixture
    def sample_labeled_docs(self):
        """Create sample labeled documents."""
        return [
            Doc(id="1", content="Fix crash on startup", category="Bug"),
            Doc(id="2", content="Fix memory leak", category="Bug"),
            Doc(id="3", content="Fix null pointer", category="Bug"),
            Doc(id="4", content="Fix seg fault", category="Bug"),
            Doc(id="5", content="Fix race condition", category="Bug"),
            Doc(id="6", content="Add dark mode", category="Feature"),
            Doc(id="7", content="Add search feature", category="Feature"),
            Doc(id="8", content="Add export feature", category="Feature"),
            Doc(id="9", content="Add import feature", category="Feature"),
            Doc(id="10", content="Add settings page", category="Feature"),
            Doc(id="11", content="Update README", category="Documentation"),
            Doc(id="12", content="Add API docs", category="Documentation"),
            Doc(id="13", content="Update tutorial", category="Documentation"),
            Doc(id="14", content="Add examples", category="Documentation"),
            Doc(id="15", content="Write changelog", category="Documentation"),
        ]

    @pytest.fixture
    def sample_embeddings(self, sample_labeled_docs):
        """Create dummy embeddings for testing."""
        # Create deterministic embeddings that somewhat reflect the categories
        np.random.seed(42)
        embeddings = []
        for doc in sample_labeled_docs:
            if doc.category == "Bug":
                # Bugs cluster around [1, 0, 0]
                base = np.array([1.0, 0.0, 0.0])
            elif doc.category == "Feature":
                # Features cluster around [0, 1, 0]
                base = np.array([0.0, 1.0, 0.0])
            else:  # Documentation
                # Docs cluster around [0, 0, 1]
                base = np.array([0.0, 0.0, 1.0])

            # Add small random noise
            noise = np.random.normal(0, 0.1, 3)
            embeddings.append((base + noise).tolist())

        return embeddings

    def test_train_classifier_basic(self, sample_labeled_docs, sample_embeddings, sample_taxonomy):
        """Test basic classifier training."""
        model, index_to_category, metrics = train_classifier(
            sample_labeled_docs, sample_embeddings, sample_taxonomy
        )

        # Check model type
        assert isinstance(model, RandomForestClassifier)

        # Check index mapping
        assert len(index_to_category) == 3
        assert "Bug" in index_to_category.values()
        assert "Feature" in index_to_category.values()

        # Check metrics exist
        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics
        assert "train_f1" in metrics
        assert "test_f1" in metrics
        assert "per_class_f1" in metrics

        # Check metrics are reasonable (0-1 range)
        assert 0 <= metrics["train_accuracy"] <= 1
        assert 0 <= metrics["test_accuracy"] <= 1

        # Check per_class_f1 has entries for the categories
        per_class = metrics["per_class_f1"]
        assert isinstance(per_class, dict)
        assert all(0 <= v <= 1 for v in per_class.values())

    def test_train_classifier_filters_invalid_categories(self, sample_embeddings, sample_taxonomy):
        """Test that classifier filters out documents with invalid categories."""
        docs_with_invalid = [
            Doc(id="1", content="Fix crash", category="Bug"),
            Doc(id="2", content="Fix leak", category="Bug"),
            Doc(id="3", content="Fix error", category="Bug"),
            Doc(id="4", content="Fix null pointer", category="Bug"),
            Doc(id="5", content="Fix seg fault", category="Bug"),
            Doc(id="6", content="Add feature", category="Feature"),
            Doc(id="7", content="Add button", category="Feature"),
            Doc(id="8", content="Add page", category="Feature"),
            Doc(id="9", content="Add menu", category="Feature"),
            Doc(id="10", content="Add dialog", category="Feature"),
            Doc(id="11", content="Unknown category", category="Other"),  # Invalid
            Doc(id="12", content="Update docs", category="Documentation"),
            Doc(id="13", content="Update guide", category="Documentation"),
            Doc(id="14", content="Update README", category="Documentation"),
            Doc(id="15", content="Update tutorial", category="Documentation"),
            Doc(id="16", content="Update examples", category="Documentation"),
        ]

        # Should train successfully, skipping the invalid category (15 valid docs)
        model, index_to_category, metrics = train_classifier(
            docs_with_invalid[:16], sample_embeddings[:16], sample_taxonomy
        )

        assert isinstance(model, RandomForestClassifier)
        assert len(index_to_category) == 3

    def test_train_classifier_no_valid_docs(self, sample_embeddings, sample_taxonomy):
        """Test that training fails when no valid documents exist."""
        docs_all_invalid = [
            Doc(id="1", content="Content", category="Invalid1"),
            Doc(id="2", content="Content", category="Invalid2"),
        ]

        with pytest.raises(ValueError, match="No valid labeled documents"):
            train_classifier(docs_all_invalid, sample_embeddings[:2], sample_taxonomy)

    def test_predict_with_classifier(self, sample_labeled_docs, sample_embeddings, sample_taxonomy):
        """Test prediction with trained classifier."""
        model, index_to_category, _ = train_classifier(
            sample_labeled_docs, sample_embeddings, sample_taxonomy
        )

        # Create new test embeddings similar to "Bug" category
        test_embeddings = [
            [0.9, 0.1, 0.0],  # Should predict Bug
            [0.0, 0.9, 0.1],  # Should predict Feature
        ]

        predictions = predict_with_classifier(model, test_embeddings, index_to_category)

        assert len(predictions) == 2
        assert all(isinstance(pred, str) for pred in predictions)
        assert all(pred in ["Bug", "Feature", "Documentation"] for pred in predictions)

    def test_get_prediction_confidence(self, sample_labeled_docs, sample_embeddings, sample_taxonomy):
        """Test getting confidence scores for predictions."""
        model, index_to_category, _ = train_classifier(
            sample_labeled_docs, sample_embeddings, sample_taxonomy
        )

        test_embeddings = [
            [1.0, 0.0, 0.0],  # Very confident Bug
            [0.5, 0.5, 0.0],  # Less confident between Bug and Feature
        ]

        confidences = get_prediction_confidence(model, test_embeddings)

        assert len(confidences) == 2
        assert all(0 <= conf <= 1 for conf in confidences)
        # First prediction should be more confident than second
        assert confidences[0] >= confidences[1]

    def test_train_classifier_handles_imbalanced_data(self, sample_taxonomy):
        """Test that classifier handles imbalanced class distribution."""
        # Create imbalanced dataset (many Bugs, few Features)
        docs = [
            Doc(id=f"bug_{i}", content=f"Bug {i}", category="Bug")
            for i in range(10)
        ] + [
            Doc(id="feat_1", content="Feature 1", category="Feature"),
            Doc(id="feat_2", content="Feature 2", category="Feature"),
            Doc(id="doc_1", content="Doc 1", category="Documentation"),
            Doc(id="doc_2", content="Doc 2", category="Documentation"),
        ]

        # Create embeddings
        np.random.seed(42)
        embeddings = [np.random.randn(5).tolist() for _ in range(len(docs))]

        # Should train successfully with class weighting
        model, index_to_category, metrics = train_classifier(docs, embeddings, sample_taxonomy)

        assert isinstance(model, RandomForestClassifier)
        # Model should use balanced class weights
        assert hasattr(model, 'class_weight')
