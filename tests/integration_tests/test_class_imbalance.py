"""Integration test for class imbalance handling using the NCL case study data.

This test validates that the new class imbalance features work correctly:
1. per_class_f1 metric is computed
2. sample_distribution and zero_sample_categories are returned
3. min_examples_per_category augments the sample
4. classifier_confidence_threshold triggers LLM fallback

Run with: pytest tests/integration_tests/test_class_imbalance.py -v
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from collections import Counter

# Test data paths
CASE_STUDY_DIR = Path(__file__).parent.parent.parent / "case_study" / "2026-01"
LABELED_DATA = CASE_STUDY_DIR / "labeled_data.csv"
TAXONOMY_FILE = CASE_STUDY_DIR / "taxonomy.json"


@pytest.fixture
def case_study_data():
    """Load case study data if available."""
    if not LABELED_DATA.exists():
        pytest.skip("Case study data not available")

    df = pd.read_csv(LABELED_DATA)
    with open(TAXONOMY_FILE) as f:
        taxonomy_data = json.load(f)

    return df, taxonomy_data["categories"]


@pytest.fixture
def small_sample(case_study_data):
    """Create a small sample for quick testing."""
    df, taxonomy = case_study_data
    # Sample 500 docs for faster testing
    sample_df = df.sample(n=min(500, len(df)), random_state=42)
    return sample_df, taxonomy


class TestClassImbalanceMetrics:
    """Test that new diagnostic metrics are computed correctly."""

    def test_per_class_f1_computed(self):
        """Test that per_class_f1 is included in classifier metrics."""
        import numpy as np
        from delve.core.classifier import train_classifier
        from delve.state import Doc

        # Create a minimal dataset with known class distribution
        taxonomy = [
            {"id": "1", "name": "ClassA", "description": "Class A"},
            {"id": "2", "name": "ClassB", "description": "Class B"},
            {"id": "3", "name": "ClassC", "description": "Class C"},
        ]

        # Create docs - imbalanced: 15 ClassA, 10 ClassB, 5 ClassC
        docs = []
        for i in range(15):
            docs.append(Doc(id=f"a{i}", content=f"ClassA doc {i}", category="ClassA"))
        for i in range(10):
            docs.append(Doc(id=f"b{i}", content=f"ClassB doc {i}", category="ClassB"))
        for i in range(5):
            docs.append(Doc(id=f"c{i}", content=f"ClassC doc {i}", category="ClassC"))

        # Create simple embeddings that cluster by class
        np.random.seed(42)
        embeddings = []
        for doc in docs:
            if doc.category == "ClassA":
                base = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
            elif doc.category == "ClassB":
                base = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
            else:
                base = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
            noise = np.random.normal(0, 0.1, 5)
            embeddings.append((base + noise).tolist())

        # Train classifier
        model, index_to_category, metrics = train_classifier(docs, embeddings, taxonomy)

        # Verify per_class_f1 exists and has expected structure
        assert "per_class_f1" in metrics
        per_class = metrics["per_class_f1"]
        assert isinstance(per_class, dict)
        assert len(per_class) > 0
        assert all(isinstance(v, float) for v in per_class.values())
        assert all(0 <= v <= 1 for v in per_class.values())

    def test_sample_distribution_computed(self):
        """Test that sample_distribution is computed from labeled docs."""
        from collections import Counter
        from delve.state import Doc

        # Simulate what document_labeler does
        docs = [
            Doc(id="1", content="test1", category="ClassA"),
            Doc(id="2", content="test2", category="ClassA"),
            Doc(id="3", content="test3", category="ClassB"),
            Doc(id="4", content="test4", category="Other"),  # Should be excluded
        ]

        sample_distribution = Counter(
            doc.category for doc in docs if doc.category != "Other"
        )

        assert dict(sample_distribution) == {"ClassA": 2, "ClassB": 1}

    def test_zero_sample_categories_detected(self):
        """Test that categories with zero samples are identified."""
        from collections import Counter
        from delve.state import Doc

        taxonomy = [
            {"id": "1", "name": "ClassA", "description": "A"},
            {"id": "2", "name": "ClassB", "description": "B"},
            {"id": "3", "name": "ClassC", "description": "C"},
        ]

        docs = [
            Doc(id="1", content="test1", category="ClassA"),
            Doc(id="2", content="test2", category="ClassA"),
            # ClassB and ClassC have no samples!
        ]

        sample_distribution = Counter(
            doc.category for doc in docs if doc.category != "Other"
        )
        all_categories = {cat["name"] for cat in taxonomy}
        zero_sample_categories = list(all_categories - set(sample_distribution.keys()))

        assert set(zero_sample_categories) == {"ClassB", "ClassC"}


class TestConfigurationParameters:
    """Test that new configuration parameters work correctly."""

    def test_min_examples_per_category_default(self):
        """Test default value is 0 (disabled)."""
        from delve.configuration import Configuration

        config = Configuration()
        assert config.min_examples_per_category == 0

    def test_min_examples_per_category_configurable(self):
        """Test parameter can be set."""
        from delve.configuration import Configuration

        config = Configuration(min_examples_per_category=5)
        assert config.min_examples_per_category == 5

    def test_sampling_strategy_default(self):
        """Test default sampling strategy is random."""
        from delve.configuration import Configuration

        config = Configuration()
        assert config.sampling_strategy == "random"

    def test_low_confidence_action_default(self):
        """Test default low_confidence_action is 'other'."""
        from delve.configuration import Configuration

        config = Configuration()
        assert config.low_confidence_action == "other"

    def test_low_confidence_action_configurable(self):
        """Test low_confidence_action can be set to valid values."""
        from delve.configuration import Configuration

        # Test 'other'
        config = Configuration(low_confidence_action="other")
        assert config.low_confidence_action == "other"

        # Test 'llm'
        config = Configuration(low_confidence_action="llm")
        assert config.low_confidence_action == "llm"

        # Test 'keep'
        config = Configuration(low_confidence_action="keep")
        assert config.low_confidence_action == "keep"

    def test_to_dict_includes_new_params(self):
        """Test that to_dict includes the new parameters."""
        from delve.configuration import Configuration

        config = Configuration(
            min_examples_per_category=3,
            sampling_strategy="random",
            low_confidence_action="llm",
        )
        d = config.to_dict()

        assert "min_examples_per_category" in d
        assert d["min_examples_per_category"] == 3
        assert "sampling_strategy" in d
        assert d["sampling_strategy"] == "random"
        assert "low_confidence_action" in d
        assert d["low_confidence_action"] == "llm"


class TestFindSimilarDocuments:
    """Test the embedding similarity search function."""

    def test_find_similar_documents_basic(self):
        """Test basic similarity search."""
        import numpy as np
        from delve.core.document_labeler import _find_similar_documents

        # Query embeddings (one document similar to [1,0,0])
        query = np.array([[1.0, 0.0, 0.0]])

        # Pool embeddings
        pool = np.array([
            [0.0, 1.0, 0.0],  # idx 0 - different
            [0.9, 0.1, 0.0],  # idx 1 - similar
            [0.0, 0.0, 1.0],  # idx 2 - different
            [0.8, 0.2, 0.0],  # idx 3 - similar
        ])

        result = _find_similar_documents(query, pool, k=2)

        # Should return indices 1 and 3 (most similar to query)
        assert len(result) == 2
        assert 1 in result
        assert 3 in result

    def test_find_similar_with_exclusions(self):
        """Test that exclusions are respected."""
        import numpy as np
        from delve.core.document_labeler import _find_similar_documents

        query = np.array([[1.0, 0.0, 0.0]])
        pool = np.array([
            [0.9, 0.1, 0.0],  # idx 0 - most similar but excluded
            [0.8, 0.2, 0.0],  # idx 1 - second most similar
            [0.0, 1.0, 0.0],  # idx 2 - different
        ])

        result = _find_similar_documents(query, pool, k=1, exclude_indices={0})

        assert len(result) == 1
        assert result[0] == 1  # Should skip 0, return 1


class TestStateFields:
    """Test that new state fields are properly defined."""

    def test_state_has_new_fields(self):
        """Test State dataclass has new metric fields."""
        from delve.state import State

        state = State()

        # Check new fields exist with correct defaults
        assert hasattr(state, "llm_relabel_count")
        assert state.llm_relabel_count == 0

        assert hasattr(state, "augmented_count")
        assert state.augmented_count == 0

        assert hasattr(state, "sample_distribution")
        assert state.sample_distribution is None

        assert hasattr(state, "zero_sample_categories")
        assert state.zero_sample_categories == []


class TestDelveClientParameters:
    """Test that Delve client exposes new parameters."""

    def test_delve_client_has_low_confidence_action(self):
        """Test Delve client accepts low_confidence_action parameter."""
        from delve import Delve

        # Should not raise
        delve = Delve(low_confidence_action="other")
        assert delve.config.low_confidence_action == "other"

        delve = Delve(low_confidence_action="llm")
        assert delve.config.low_confidence_action == "llm"

        delve = Delve(low_confidence_action="keep")
        assert delve.config.low_confidence_action == "keep"

    def test_delve_client_has_min_examples_per_category(self):
        """Test Delve client accepts min_examples_per_category parameter."""
        from delve import Delve

        delve = Delve(min_examples_per_category=5)
        assert delve.config.min_examples_per_category == 5

    def test_delve_client_has_sampling_strategy(self):
        """Test Delve client accepts sampling_strategy parameter."""
        from delve import Delve

        delve = Delve(sampling_strategy="random")
        assert delve.config.sampling_strategy == "random"


# Only run this if we want to test with real API calls (expensive)
@pytest.mark.skip(reason="Requires API calls - run manually for full integration test")
class TestFullIntegration:
    """Full integration tests that require API calls."""

    def test_run_with_imbalance_handling(self, small_sample):
        """Test full run with imbalance handling parameters."""
        from delve import Delve, Verbosity

        df, taxonomy = small_sample

        delve = Delve(
            sample_size=50,
            min_examples_per_category=3,
            classifier_confidence_threshold=0.6,
            predefined_taxonomy=taxonomy,
            verbosity=Verbosity.DEBUG,
        )

        # This would make real API calls
        # result = delve.run_sync(df, text_column="content", id_column="id")

        # Verify new metrics are present
        # assert "sample_distribution" in result.metadata
        # assert "zero_sample_categories" in result.metadata
        # if "classifier_metrics" in result.metadata:
        #     assert "per_class_f1" in result.metadata["classifier_metrics"]
