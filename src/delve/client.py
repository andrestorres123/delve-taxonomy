"""Main SDK client for Delve taxonomy generation."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import pandas as pd

from delve.configuration import Configuration
from delve.console import Console, Verbosity
from delve.result import DelveResult
from delve.adapters import create_adapter
from delve.graph import graph
from delve.state import State, Doc
from delve.utils import validate_all_api_keys


class Delve:
    """Main client for Delve taxonomy generation.

    This class provides a simple interface for generating taxonomies
    from various data sources and exporting results.

    Examples:
        >>> # Basic CSV usage
        >>> delve = Delve()
        >>> result = delve.run_sync("data.csv", text_column="text")
        >>> print(f"Generated {len(result.taxonomy)} categories")

        >>> # With custom configuration
        >>> delve = Delve(
        ...     model="anthropic/claude-sonnet-4-5-20250929",
        ...     sample_size=200,
        ...     output_dir="./my_results"
        ... )
        >>> result = delve.run_sync("data.json", text_field="content")
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-5-20250929",
        fast_llm: Optional[str] = None,
        sample_size: int = 100,
        batch_size: int = 200,
        use_case: Optional[str] = None,
        output_dir: str = "./results",
        output_formats: Optional[List[str]] = None,
        verbosity: Verbosity = Verbosity.SILENT,
        console: Optional[Console] = None,
        predefined_taxonomy: Optional[Union[str, List[Dict[str, str]]]] = None,
        embedding_model: str = "text-embedding-3-large",
        classifier_confidence_threshold: float = 0.0,
        low_confidence_action: str = "other",
        max_num_clusters: int = 5,
        min_examples_per_category: int = 0,
        sampling_strategy: str = "random",
    ):
        """Initialize Delve client.

        Args:
            model: Main LLM model for reasoning (default: Claude 3.5 Sonnet)
            fast_llm: Faster model for summarization (default: Claude 3 Haiku)
            sample_size: Number of documents to sample for LLM labeling.
                If sample_size < total documents, trains a classifier to label the rest efficiently.
                Set to 0 to process all documents.
            batch_size: Batch size for minibatch processing
            use_case: Description of the taxonomy use case
            output_dir: Directory for output files
            output_formats: List of formats to generate (json, csv, markdown)
            verbosity: Verbosity level (SILENT, QUIET, NORMAL, VERBOSE, DEBUG).
                SDK default is SILENT. Use NORMAL for progress output.
            console: Optional Console instance. If not provided, one is created
                based on verbosity level.
            predefined_taxonomy: Pre-defined taxonomy to use instead of discovery.
                Can be a file path (JSON/CSV) or a list of dicts with 'id', 'name', 'description'.
                When provided, skips the discovery phase and directly labels documents.
            embedding_model: OpenAI embedding model for classifier training (default: text-embedding-3-large)
            classifier_confidence_threshold: Minimum confidence for classifier predictions.
                Documents below threshold are handled by low_confidence_action (default: 0.0 = disabled).
            low_confidence_action: Action for low-confidence predictions: 'other' (label as Other),
                'llm' (re-label with LLM, max 20 docs), or 'keep' (keep classifier prediction).
                Default is 'other'.
            max_num_clusters: Maximum number of clusters/categories to generate (default: 5).
            min_examples_per_category: Minimum training examples per category.
                If a category has fewer samples, Delve will find more via embedding similarity.
                Set to 0 to disable (default).
            sampling_strategy: Sampling strategy: 'random' (default) or 'stratified'.
        """
        self.config = Configuration(
            model=model,
            fast_llm=fast_llm or "anthropic/claude-haiku-4-5-20251001",
            sample_size=sample_size,
            batch_size=batch_size,
            use_case=use_case or "Generate taxonomy for categorizing document content",
            output_dir=output_dir,
            output_formats=output_formats or ["json", "csv", "markdown"],
            verbosity=verbosity,
            console=console,
            predefined_taxonomy=predefined_taxonomy,
            embedding_model=embedding_model,
            classifier_confidence_threshold=classifier_confidence_threshold,
            low_confidence_action=low_confidence_action,
            max_num_clusters=max_num_clusters,
            min_examples_per_category=min_examples_per_category,
            sampling_strategy=sampling_strategy,
        )
        self.console = self.config.get_console()

    async def run_with_docs(
        self,
        docs: List[Doc],
    ) -> DelveResult:
        """Run taxonomy generation on pre-created Doc objects.

        Use this method when you already have Doc objects (e.g., for testing
        or when creating docs programmatically).

        Args:
            docs: List of Doc objects to process

        Returns:
            DelveResult: Results object with taxonomy and labeled documents

        Examples:
            >>> from delve import Delve, Doc
            >>> docs = [
            ...     Doc(id="1", content="Fix authentication bug"),
            ...     Doc(id="2", content="Add dark mode feature"),
            ... ]
            >>> delve = Delve(use_case="Categorize software issues")
            >>> result = await delve.run_with_docs(docs)
        """
        # Start timing
        start_time = time.time()

        # Validate API keys early
        # OpenAI key needed if sample_size > 0 (classifier uses embeddings)
        needs_openai = self.config.sample_size > 0 and len(docs) > self.config.sample_size
        try:
            validate_all_api_keys(needs_openai=needs_openai)
        except ValueError as e:
            self.console.error(str(e))
            raise

        # Create initial state with docs
        initial_state = State(all_documents=docs)

        # Run the graph with status spinner
        with self.console.status(f"Processing {len(docs)} documents..."):
            result_state = await graph.ainvoke(
                initial_state,
                config={"configurable": self.config.to_dict()},
            )

        # Calculate run duration
        run_duration = time.time() - start_time

        # Source info for Doc-based input
        source_info: Dict[str, Any] = {
            "type": "docs",
            "path": None,
            "text_column": None,
            "id_column": None,
        }

        # Create result object
        delve_result = DelveResult.from_state(
            result_state,
            self.config,
            run_duration=run_duration,
            source_info=source_info,
        )

        self.console.success(f"Generated {len(delve_result.taxonomy)} categories")
        self.console.success(f"Labeled {len(delve_result.labeled_documents)} documents")
        self.console.success(f"Results saved to {self.config.output_dir}/")

        return delve_result

    def run_with_docs_sync(
        self,
        docs: List[Doc],
    ) -> DelveResult:
        """Synchronous wrapper for run_with_docs().

        Args:
            docs: List of Doc objects to process

        Returns:
            DelveResult: Results object with taxonomy and labeled documents

        Examples:
            >>> from delve import Delve, Doc
            >>> docs = [Doc(id="1", content="Fix bug"), ...]
            >>> delve = Delve()
            >>> result = delve.run_with_docs_sync(docs)
        """
        # Check if we're in a Jupyter/Colab environment with existing event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, use asyncio.run()
            return asyncio.run(self.run_with_docs(docs))
        else:
            # Event loop is already running (Jupyter/Colab)
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Cannot use run_with_docs_sync() in Jupyter/Colab without nest_asyncio. "
                    "Install it with: !pip install nest-asyncio\n"
                    "Then import and apply it at the top of your notebook:\n"
                    "  import nest_asyncio\n"
                    "  nest_asyncio.apply()\n"
                    "Alternatively, use the async version:\n"
                    "  result = await delve_client.run_with_docs(...)"
                )

            # nest_asyncio is available, run normally
            return asyncio.run(self.run_with_docs(docs))

    async def run(
        self,
        data: Union[str, Path, pd.DataFrame],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        source_type: Optional[str] = None,
        **adapter_kwargs,
    ) -> DelveResult:
        """Run taxonomy generation on data.

        Args:
            data: Data source (file path, URI, or DataFrame)
            text_column: Column/field containing text content
            id_column: Optional column/field for document IDs
            source_type: Force specific adapter type (csv, json, langsmith, dataframe)
            **adapter_kwargs: Additional adapter-specific parameters
                - For JSON: json_path, text_field
                - For LangSmith: api_key, days, max_runs, filter_expr

        Returns:
            DelveResult: Results object with taxonomy and labeled documents

        Raises:
            ValueError: If data source is invalid or required parameters are missing
            Exception: If taxonomy generation fails

        Examples:
            >>> # CSV file
            >>> result = await delve.run("data.csv", text_column="text")

            >>> # JSON with JSONPath
            >>> result = await delve.run(
            ...     "data.json",
            ...     json_path="$.messages[*].content"
            ... )

            >>> # LangSmith
            >>> result = await delve.run(
            ...     "langsmith://my-project",
            ...     api_key="lsv2_...",
            ...     days=7
            ... )
        """
        # Start timing
        start_time = time.time()

        # Debug: Show full configuration
        self.console.debug("=" * 50)
        self.console.debug("Delve Configuration:")
        self.console.debug(f"  Model: {self.config.model}")
        self.console.debug(f"  Fast LLM: {self.config.fast_llm}")
        self.console.debug(f"  Sample size: {self.config.sample_size}")
        self.console.debug(f"  Batch size: {self.config.batch_size}")
        self.console.debug(f"  Max clusters: {self.config.max_num_clusters}")
        self.console.debug(f"  Embedding model: {self.config.embedding_model}")
        self.console.debug(f"  Output dir: {self.config.output_dir}")
        self.console.debug(f"  Use case: {self.config.use_case}")
        self.console.debug("=" * 50)

        # 0. Validate API keys before starting
        # Check for OpenAI key if sample_size > 0 (might need embeddings for classifier)
        # We check conservatively since we don't know doc count yet
        try:
            validate_all_api_keys(needs_openai=(self.config.sample_size > 0))
        except ValueError as e:
            self.console.error(str(e))
            raise

        # 1. Create adapter and load data
        # Filter out configuration parameters that shouldn't be passed to adapters
        config_params = {
            "model", "fast_llm", "sample_size", "batch_size",
            "output_formats", "output_dir", "verbosity", "use_case"
        }
        filtered_kwargs = {
            k: v for k, v in adapter_kwargs.items()
            if k not in config_params
        }

        with self.console.status(f"Loading data from {data}..."):
            adapter = create_adapter(
                data,
                text_column=text_column,
                id_column=id_column,
                source_type=source_type,
                **filtered_kwargs,
            )

            # Validate and load documents
            adapter.validate()
            documents = await adapter.load()

        self.console.success(f"Loaded {len(documents)} documents")

        # 2. Run graph with documents in initial state
        status_msg = (
            "Using predefined taxonomy to label documents..."
            if self.config.predefined_taxonomy
            else "Generating taxonomy..."
        )

        initial_state = {
            "all_documents": documents,
        }

        with self.console.status(status_msg):
            result_state = await graph.ainvoke(
                initial_state,
                config={"configurable": self.config.to_dict()},
            )

        if self.config.predefined_taxonomy:
            self.console.success("Document labeling complete")
        else:
            self.console.success("Taxonomy generation complete")

        # Calculate run duration
        run_duration = time.time() - start_time

        # Build source info
        source_info: Dict[str, Any] = {
            "type": source_type or "auto",
            "path": str(data) if not isinstance(data, pd.DataFrame) else None,
            "text_column": text_column,
            "id_column": id_column,
        }

        # 3. Create result object with extra metadata
        delve_result = DelveResult.from_state(
            result_state,
            self.config,
            run_duration=run_duration,
            source_info=source_info,
        )

        # 4. Export is handled by save_results node in the graph
        self.console.success(f"Results saved to {self.config.output_dir}/")

        return delve_result

    def run_sync(
        self,
        data: Union[str, Path, pd.DataFrame],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        source_type: Optional[str] = None,
        **adapter_kwargs,
    ) -> DelveResult:
        """Synchronous wrapper for run().

        This is a convenience method for users who don't want to deal
        with async/await syntax. Works in Jupyter/Colab environments.

        Args:
            data: Data source (file path, URI, or DataFrame)
            text_column: Column/field containing text content
            id_column: Optional column/field for document IDs
            source_type: Force specific adapter type
            **adapter_kwargs: Additional adapter-specific parameters

        Returns:
            DelveResult: Results object with taxonomy and labeled documents

        Examples:
            >>> delve = Delve()
            >>> result = delve.run_sync("data.csv", text_column="text")
            >>> print(result.taxonomy)
        """
        # Check if we're in a Jupyter/Colab environment with existing event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, use asyncio.run()
            return asyncio.run(
                self.run(
                    data,
                    text_column=text_column,
                    id_column=id_column,
                    source_type=source_type,
                    **adapter_kwargs,
                )
            )
        else:
            # Event loop is already running (Jupyter/Colab)
            # Try to use nest_asyncio to allow nested event loops
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Cannot use run_sync() in Jupyter/Colab without nest_asyncio. "
                    "Install it with: !pip install nest-asyncio\n"
                    "Then import and apply it at the top of your notebook:\n"
                    "  import nest_asyncio\n"
                    "  nest_asyncio.apply()\n"
                    "Alternatively, use the async version:\n"
                    "  result = await delve_client.run(...)"
                )

            # nest_asyncio is available, run normally
            return asyncio.run(
                self.run(
                    data,
                    text_column=text_column,
                    id_column=id_column,
                    source_type=source_type,
                    **adapter_kwargs,
                )
            )
