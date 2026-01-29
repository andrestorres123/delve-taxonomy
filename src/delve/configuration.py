"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Annotated, Optional, List, Union, Dict

from langchain_core.runnables import RunnableConfig, ensure_config

from delve.console import Console, NullConsole, Verbosity

if TYPE_CHECKING:
    pass


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the Delve taxonomy generator."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-sonnet-4-5-20250929",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    fast_llm: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-haiku-4-5-20251001",
        metadata={
            "description": "A faster, lighter model for tasks like summarization. "
            "Should be in the form: provider/model-name."
        },
    )

    sample_size: int = field(
        default=100,
        metadata={
            "description": "Number of documents to sample for processing."
        },
    )

    batch_size: int = field(
        default=200,
        metadata={
            "description": "Size of minibatches for document processing."
        },
    )

    output_formats: List[str] = field(
        default_factory=lambda: ["json", "csv", "markdown"],
        metadata={
            "description": "List of output formats to generate (json, csv, markdown)."
        },
    )

    output_dir: str = field(
        default="./results",
        metadata={
            "description": "Directory where output files will be saved."
        },
    )

    verbosity: Verbosity = field(
        default=Verbosity.SILENT,
        metadata={
            "description": "Verbosity level: SILENT (0), QUIET (1), NORMAL (2), VERBOSE (3), DEBUG (4)."
        },
    )

    console: Optional[Console] = field(
        default=None,
        repr=False,
        metadata={
            "description": "Console instance for output. Not serialized."
        },
    )

    use_case: str = field(
        default="Generate taxonomy for categorizing document content",
        metadata={
            "description": "Description of the taxonomy use case."
        },
    )

    predefined_taxonomy: Optional[Union[str, List[Dict[str, str]]]] = field(
        default=None,
        metadata={
            "description": "Pre-defined taxonomy to use instead of discovering one. "
            "Can be a file path (JSON/CSV) or a list of category dicts with 'id', 'name', 'description'."
        },
    )

    embedding_model: str = field(
        default="text-embedding-3-large",
        metadata={
            "description": "OpenAI embedding model to use for classifier training. "
            "Used when sample_size < total documents to train efficient classifier."
        },
    )

    classifier_confidence_threshold: float = field(
        default=0.0,
        metadata={
            "description": "Minimum confidence threshold for classifier predictions. "
            "Documents below this threshold will be handled according to low_confidence_action. "
            "Set to 0 to disable (default)."
        },
    )

    low_confidence_action: str = field(
        default="other",
        metadata={
            "description": "Action for low-confidence predictions: 'other' (label as Other), "
            "'llm' (re-label with LLM, max 20 docs), or 'keep' (keep classifier prediction). "
            "Default is 'other'."
        },
    )

    max_num_clusters: int = field(
        default=5,
        metadata={
            "description": "Maximum number of clusters/categories to generate in the taxonomy."
        },
    )

    min_examples_per_category: int = field(
        default=0,
        metadata={
            "description": "Minimum training examples per category. "
            "If a category has fewer samples, Delve will find more via embedding similarity. "
            "Set to 0 to disable (default)."
        },
    )

    sampling_strategy: str = field(
        default="random",
        metadata={
            "description": "Sampling strategy: 'random' (default) or 'stratified'. "
            "Stratified sampling balances the training set across categories."
        },
    )

    def __post_init__(self) -> None:
        """Initialize console based on verbosity."""
        # Create console if not provided
        if self.console is None:
            self.console = Console(self.verbosity)

    def get_console(self) -> Console:
        """Get the console instance, creating one if needed.

        Returns:
            Console instance for output.
        """
        if self.console is None:
            self.console = Console(self.verbosity)
        return self.console

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}

        # Filter to valid fields, excluding console (handled separately)
        _fields = {f.name for f in fields(cls) if f.init and f.name != "console"}
        init_kwargs = {k: v for k, v in configurable.items() if k in _fields}

        # Handle verbosity conversion from int if needed
        if "verbosity" in init_kwargs and isinstance(init_kwargs["verbosity"], int):
            init_kwargs["verbosity"] = Verbosity(init_kwargs["verbosity"])

        # Extract console from config if present
        console = configurable.get("console")

        return cls(console=console, **init_kwargs)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for SDK usage.

        Note: console is included to pass through to graph nodes.
        """
        return {
            "model": self.model,
            "fast_llm": self.fast_llm,
            "sample_size": self.sample_size,
            "batch_size": self.batch_size,
            "output_formats": self.output_formats,
            "output_dir": self.output_dir,
            "verbosity": self.verbosity,
            "use_case": self.use_case,
            "predefined_taxonomy": self.predefined_taxonomy,
            "embedding_model": self.embedding_model,
            "classifier_confidence_threshold": self.classifier_confidence_threshold,
            "low_confidence_action": self.low_confidence_action,
            "max_num_clusters": self.max_num_clusters,
            "min_examples_per_category": self.min_examples_per_category,
            "sampling_strategy": self.sampling_strategy,
            "console": self.console,
        }
