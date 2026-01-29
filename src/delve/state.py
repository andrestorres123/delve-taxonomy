"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional, Dict
import operator

from langgraph.managed import IsLastStep


@dataclass
class Doc:
    """Represents a document in the taxonomy generation process."""
    id: str
    content: str
    summary: Optional[str] = None
    explanation: Optional[str] = None
    category: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class InputState:
    """Defines the input state for the agent.

    In the non-interactive version, data loading is handled by adapters
    and documents are passed in via all_documents field.
    Configuration is passed separately.
    """
    all_documents: List[Doc] = field(default_factory=list)


@dataclass
class OutputState:
    """Defines the output state for the agent."""
    status: Annotated[List[str], operator.add] = field(default_factory=list)


@dataclass
class State(InputState, OutputState):
    """Represents the complete state of the taxonomy generation agent.

    This class contains all attributes needed throughout the taxonomy
    generation process.
    """
    documents: List[Doc] = field(default_factory=list)
    minibatches: List[List[int]] = field(default_factory=list)
    clusters: Annotated[List[List[Dict]], operator.add] = field(default_factory=list)
    # Note: status is inherited from OutputState, don't redefine it here
    use_case: str = field(default="")
    is_last_step: IsLastStep = field(default=False)

    # Metadata tracking
    classifier_metrics: Optional[Dict[str, float]] = None
    llm_labeled_count: int = 0
    classifier_labeled_count: int = 0
    llm_relabel_count: int = 0
    augmented_count: int = 0
    skipped_document_count: int = 0
    warnings: List[str] = field(default_factory=list)
    sample_distribution: Optional[Dict[str, int]] = None
    zero_sample_categories: List[str] = field(default_factory=list)

    # Classifier storage for export
    classifier_model: Optional[Any] = None  # RandomForestClassifier
    classifier_index_to_category: Optional[Dict[int, str]] = None
