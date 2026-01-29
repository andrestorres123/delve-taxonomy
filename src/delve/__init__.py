"""Delve: AI-powered taxonomy generation for your data."""

from delve.client import Delve
from delve.console import Console, Verbosity
from delve.state import Doc
from delve.configuration import Configuration
from delve.result import DelveResult, TaxonomyCategory, ClassificationResult, TrainingResult, MatchResult
from delve.core.classifier import ClassifierBundle

__version__ = "0.1.12"

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
