"""Embedding-based classifier for document labeling at scale."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

from delve.state import Doc

if TYPE_CHECKING:
    from delve.console import Console


@dataclass
class ClassifierBundle:
    """Bundle containing a trained classifier and all metadata needed for inference.

    This bundle can be serialized to disk with joblib and loaded later to
    classify new documents without any LLM calls.
    """
    model: RandomForestClassifier
    index_to_category: Dict[int, str]
    embedding_model: str
    embedding_dimensions: int
    taxonomy: List[Dict[str, str]]
    metrics: Dict[str, Any]
    created_at: str
    delve_version: str


def save_bundle(bundle: ClassifierBundle, path: Union[str, Path]) -> Path:
    """Save classifier bundle to joblib file.

    Args:
        bundle: ClassifierBundle to save
        path: Output file path (should end with .joblib)

    Returns:
        Path to saved file
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Union[str, Path]) -> ClassifierBundle:
    """Load classifier bundle from joblib file.

    Args:
        path: Path to saved classifier bundle

    Returns:
        ClassifierBundle instance

    Raises:
        ValueError: If file doesn't contain a valid ClassifierBundle
        FileNotFoundError: If file doesn't exist
    """
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier bundle not found: {path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, ClassifierBundle):
        raise ValueError(
            f"Invalid classifier bundle format. Expected ClassifierBundle, "
            f"got {type(bundle).__name__}"
        )
    return bundle


def _infer_taxonomy_from_labels(labels: List[str]) -> List[Dict[str, str]]:
    """Create minimal taxonomy from unique labels.

    Args:
        labels: List of category labels

    Returns:
        List of taxonomy category dicts with id, name, description
    """
    unique_labels = sorted(set(labels))
    return [
        {
            "id": str(i + 1),
            "name": label,
            "description": f"Category: {label}",
        }
        for i, label in enumerate(unique_labels)
    ]


def train_classifier(
    labeled_docs: List[Doc],
    embeddings: List[List[float]],
    taxonomy: List[Dict[str, str]],
    console: Optional["Console"] = None,
) -> Tuple[RandomForestClassifier, Dict[int, str], Dict[str, float]]:
    """Train a RandomForest classifier on labeled documents.

    Args:
        labeled_docs: Documents with LLM-assigned categories
        embeddings: Embeddings for the labeled documents
        taxonomy: List of category dicts with 'id', 'name', 'description'
        console: Optional Console instance for output

    Returns:
        Tuple of (trained model, index_to_category mapping, metrics dict)
    """
    # Create mappings
    category_to_index = {cat["name"]: i for i, cat in enumerate(taxonomy)}
    index_to_category = {i: cat["name"] for i, cat in enumerate(taxonomy)}

    # Prepare training data
    X = np.array(embeddings)
    y = []
    for doc in labeled_docs:
        if doc.category not in category_to_index:
            # Skip documents with invalid categories (e.g., "Other")
            available = list(category_to_index.keys())
            if console:
                console.warning(
                    f"Skipping document with category '{doc.category}' "
                    f"not in taxonomy: {available}"
                )
            continue
        y.append(category_to_index[doc.category])

    if len(y) == 0:
        raise ValueError("No valid labeled documents to train classifier")

    y = np.array(y)

    # Filter X to match y (remove skipped documents)
    if len(y) < len(embeddings):
        valid_indices = [i for i, doc in enumerate(labeled_docs) if doc.category in category_to_index]
        X = np.array([embeddings[i] for i in valid_indices])

    # Check if we can do stratified splitting
    # We need at least 2 samples per class for stratification
    unique, counts = np.unique(y, return_counts=True)
    min_samples_per_class = counts.min()
    can_stratify = min_samples_per_class >= 2 and len(np.unique(y)) > 1

    # Debug: Show class distribution
    if console:
        console.debug(f"Training set: {len(y)} samples, {len(unique)} classes")
        console.debug(f"Class distribution:")
        for cls_idx, count in zip(unique, counts):
            cls_name = index_to_category.get(cls_idx, f"Unknown({cls_idx})")
            console.debug(f"  [{cls_idx}] {cls_name}: {count} samples")
        console.debug(f"Stratified split: {can_stratify}")

    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )

    # Calculate class weights to handle imbalanced data
    # Note: np.unique returns sorted unique values, and compute_class_weight
    # returns weights in the same order. We must map actual class indices
    # (not 0,1,2...) to weights since some categories may be missing.
    unique_classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train
    )
    class_weight_dict = dict(zip(unique_classes, class_weights))

    # Train RandomForest
    model = RandomForestClassifier(
        class_weight=class_weight_dict,
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_preds),
        "test_accuracy": accuracy_score(y_test, test_preds),
        "train_f1": f1_score(y_train, train_preds, average="weighted"),
        "test_f1": f1_score(y_test, test_preds, average="weighted"),
    }

    # Add per-class F1 scores for diagnosing imbalance issues
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    per_class_f1 = {
        index_to_category[int(k)]: v['f1-score']
        for k, v in report.items()
        if k.isdigit()
    }
    metrics["per_class_f1"] = per_class_f1

    return model, index_to_category, metrics


def predict_with_classifier(
    model: RandomForestClassifier,
    embeddings: List[List[float]],
    index_to_category: Dict[int, str],
) -> List[str]:
    """Predict categories using the trained classifier.

    Args:
        model: Trained RandomForest classifier
        embeddings: Document embeddings
        index_to_category: Mapping from class index to category name

    Returns:
        List of predicted category names
    """
    X = np.array(embeddings)
    predictions = model.predict(X)
    return [index_to_category[pred] for pred in predictions]


def get_prediction_confidence(
    model: RandomForestClassifier,
    embeddings: List[List[float]],
) -> List[float]:
    """Get confidence scores for predictions.

    Args:
        model: Trained RandomForest classifier
        embeddings: Document embeddings

    Returns:
        List of confidence scores (max probability for each prediction)
    """
    X = np.array(embeddings)
    probabilities = model.predict_proba(X)
    return [float(np.max(probs)) for probs in probabilities]
