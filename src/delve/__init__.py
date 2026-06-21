"""Delve: AI-powered taxonomy generation for your data."""

from delve.client import Delve
from delve.configuration import Configuration
from delve.console import Console, Verbosity
from delve.core.classifier import ClassifierBundle
from delve.result import (
    ClassificationResult,
    DelveResult,
    MatchResult,
    TaxonomyCategory,
    TrainingResult,
)
from delve.state import Doc

__version__ = "0.2.1"

__all__ = [
    "Delve",
    "Console",
    "Verbosity",
    "Doc",
    "Configuration",
    "DelveResult",
    "TaxonomyCategory",
    "ClassificationResult",
    "TrainingResult",
    "MatchResult",
    "ClassifierBundle",
    "__version__",
]
