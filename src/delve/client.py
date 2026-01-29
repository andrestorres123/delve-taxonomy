"""Main SDK client for Delve taxonomy generation."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import pandas as pd

from delve.configuration import Configuration
from delve.console import Console, Verbosity
from delve.result import (
    DelveResult,
    ClassificationResult,
    TrainingResult,
    TaxonomyCategory,
    MatchResult,
)
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

    # =========================================================================
    # Class Methods for Classifier Operations
    # =========================================================================

    @classmethod
    async def classify_async(
        cls,
        data: Union[str, Path, pd.DataFrame, List[Doc]],
        classifier_path: Union[str, Path],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        include_confidence: bool = True,
        verbosity: Verbosity = Verbosity.SILENT,
    ) -> ClassificationResult:
        """Classify documents using a saved classifier.

        This method loads a previously saved classifier bundle and uses it
        to classify new documents. Only embedding API calls are made - no
        LLM costs.

        Args:
            data: Documents to classify. Can be:
                - Path to CSV/JSON file
                - pandas DataFrame
                - List of Doc objects
            classifier_path: Path to saved classifier bundle (.joblib)
            text_column: Column containing text (required for CSV/DataFrame)
            id_column: Optional column for document IDs
            include_confidence: Whether to include confidence scores (default: True)
            verbosity: Output verbosity level

        Returns:
            ClassificationResult with classified documents

        Example:
            >>> predictions = await Delve.classify_async(
            ...     "new_data.csv",
            ...     classifier_path="classifier.joblib",
            ...     text_column="text",
            ... )
            >>> for doc in predictions.documents:
            ...     print(f"{doc.id}: {doc.category} ({doc.confidence:.2%})")
        """
        from langchain_openai import OpenAIEmbeddings
        from delve.core.classifier import (
            load_bundle,
            predict_with_classifier,
            get_prediction_confidence,
        )

        console = Console(verbosity)

        # Load classifier bundle
        with console.status(f"Loading classifier from {classifier_path}..."):
            bundle = load_bundle(classifier_path)

        console.info(
            f"Loaded classifier trained on {bundle.embedding_model} "
            f"with {len(bundle.taxonomy)} categories"
        )

        # Load documents
        docs = cls._load_docs_for_classification(data, text_column, id_column)
        console.info(f"Loaded {len(docs)} documents to classify")

        # Generate embeddings
        with console.status(f"Generating embeddings for {len(docs)} documents..."):
            encoder = OpenAIEmbeddings(model=bundle.embedding_model)
            embeddings = await encoder.aembed_documents([d.content for d in docs])

        # Predict categories
        with console.status("Classifying documents..."):
            predictions = predict_with_classifier(
                bundle.model, embeddings, bundle.index_to_category
            )

            if include_confidence:
                confidences = get_prediction_confidence(bundle.model, embeddings)
            else:
                confidences = [None] * len(docs)

        # Update docs with predictions
        for doc, category, confidence in zip(docs, predictions, confidences):
            doc.category = category
            doc.confidence = confidence

        console.success(f"Classified {len(docs)} documents")

        return ClassificationResult(
            documents=docs,
            classifier_info={
                "classifier_path": str(classifier_path),
                "embedding_model": bundle.embedding_model,
                "num_categories": len(bundle.taxonomy),
                "taxonomy": bundle.taxonomy,
                "created_at": bundle.created_at,
                "delve_version": bundle.delve_version,
                "original_metrics": bundle.metrics,
            },
        )

    @classmethod
    def classify(
        cls,
        data: Union[str, Path, pd.DataFrame, List[Doc]],
        classifier_path: Union[str, Path],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        include_confidence: bool = True,
        verbosity: Verbosity = Verbosity.SILENT,
    ) -> ClassificationResult:
        """Classify documents using a saved classifier (sync version).

        See classify_async() for full documentation.

        Example:
            >>> predictions = Delve.classify(
            ...     "new_data.csv",
            ...     classifier_path="classifier.joblib",
            ...     text_column="text",
            ... )
            >>> print(predictions.to_dataframe())
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                cls.classify_async(
                    data,
                    classifier_path,
                    text_column=text_column,
                    id_column=id_column,
                    include_confidence=include_confidence,
                    verbosity=verbosity,
                )
            )
        else:
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Cannot use classify() in Jupyter/Colab without nest_asyncio. "
                    "Use classify_async() instead or install nest_asyncio."
                )
            return asyncio.run(
                cls.classify_async(
                    data,
                    classifier_path,
                    text_column=text_column,
                    id_column=id_column,
                    include_confidence=include_confidence,
                    verbosity=verbosity,
                )
            )

    @classmethod
    async def train_from_labeled_async(
        cls,
        data: Union[str, Path, pd.DataFrame],
        text_column: str,
        label_column: str,
        id_column: Optional[str] = None,
        taxonomy: Optional[Union[str, Path, List[Dict[str, str]]]] = None,
        embedding_model: str = "text-embedding-3-large",
        test_size: float = 0.2,
        verbosity: Verbosity = Verbosity.SILENT,
    ) -> TrainingResult:
        """Train a classifier from a labeled dataset.

        Use this to train a classifier from your own labeled data (or
        corrected Delve output) without any LLM calls during training.

        Args:
            data: Labeled data source. Can be:
                - Path to CSV/JSON file
                - pandas DataFrame
            text_column: Column containing document text
            label_column: Column containing category labels
            id_column: Optional column for document IDs
            taxonomy: Optional explicit taxonomy with descriptions. Can be:
                - Path to JSON/CSV file
                - List of dicts with 'id', 'name', 'description'
                If not provided, taxonomy is inferred from unique labels.
            embedding_model: OpenAI embedding model (default: text-embedding-3-large)
            test_size: Fraction of data for validation (default: 0.2)
            verbosity: Output verbosity level

        Returns:
            TrainingResult with trained model and metrics

        Example:
            >>> result = await Delve.train_from_labeled_async(
            ...     "labeled_data.csv",
            ...     text_column="text",
            ...     label_column="category",
            ... )
            >>> print(f"Test F1: {result.metrics['test_f1']:.2%}")
            >>> result.save_classifier("my_classifier.joblib")
        """
        from langchain_openai import OpenAIEmbeddings
        from delve.core.classifier import train_classifier, _infer_taxonomy_from_labels
        from delve.core.data_loader import _load_predefined_taxonomy

        console = Console(verbosity)

        # Load data
        df = cls._load_dataframe(data)
        console.info(f"Loaded {len(df)} labeled documents")

        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found. Available: {list(df.columns)}")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns)}")

        # Get labels
        labels = df[label_column].astype(str).tolist()
        unique_labels = set(labels)
        console.info(f"Found {len(unique_labels)} unique categories")

        # Build taxonomy
        if taxonomy is not None:
            taxonomy_list = _load_predefined_taxonomy(taxonomy)
            taxonomy_names = {cat["name"] for cat in taxonomy_list}
            # Validate all labels are in taxonomy
            missing = unique_labels - taxonomy_names
            if missing:
                raise ValueError(
                    f"Labels not in taxonomy: {missing}. "
                    f"Available categories: {taxonomy_names}"
                )
        else:
            taxonomy_list = _infer_taxonomy_from_labels(labels)
            console.info("Inferred taxonomy from labels")

        # Create Doc objects
        docs = []
        for idx, row in df.iterrows():
            doc_id = str(row[id_column]) if id_column and id_column in df.columns else str(idx)
            docs.append(Doc(
                id=doc_id,
                content=str(row[text_column]),
                category=str(row[label_column]),
            ))

        # Generate embeddings
        with console.status(f"Generating embeddings for {len(docs)} documents..."):
            encoder = OpenAIEmbeddings(model=embedding_model)
            embeddings = await encoder.aembed_documents([d.content for d in docs])

        # Train classifier
        with console.status("Training classifier..."):
            model, index_to_category, metrics = train_classifier(
                docs,
                embeddings,
                taxonomy_list,
                console=console,
            )

        console.success(
            f"Classifier trained - Test F1: {metrics['test_f1']:.3f}, "
            f"Test Accuracy: {metrics['test_accuracy']:.3f}"
        )

        # Calculate counts
        total = len(docs)
        train_count = int(total * (1 - test_size))
        val_count = total - train_count

        return TrainingResult(
            model=model,
            index_to_category=index_to_category,
            taxonomy=[
                TaxonomyCategory(
                    id=cat["id"],
                    name=cat["name"],
                    description=cat["description"],
                )
                for cat in taxonomy_list
            ],
            metrics=metrics,
            training_docs_count=train_count,
            validation_docs_count=val_count,
            embedding_model=embedding_model,
            created_at=datetime.now().isoformat(),
        )

    @classmethod
    def train_from_labeled(
        cls,
        data: Union[str, Path, pd.DataFrame],
        text_column: str,
        label_column: str,
        id_column: Optional[str] = None,
        taxonomy: Optional[Union[str, Path, List[Dict[str, str]]]] = None,
        embedding_model: str = "text-embedding-3-large",
        test_size: float = 0.2,
        verbosity: Verbosity = Verbosity.SILENT,
    ) -> TrainingResult:
        """Train a classifier from a labeled dataset (sync version).

        See train_from_labeled_async() for full documentation.

        Example:
            >>> result = Delve.train_from_labeled(
            ...     "corrected_labels.csv",
            ...     text_column="text",
            ...     label_column="category",
            ... )
            >>> result.save_classifier("improved_classifier.joblib")
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                cls.train_from_labeled_async(
                    data,
                    text_column,
                    label_column,
                    id_column=id_column,
                    taxonomy=taxonomy,
                    embedding_model=embedding_model,
                    test_size=test_size,
                    verbosity=verbosity,
                )
            )
        else:
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Cannot use train_from_labeled() in Jupyter/Colab without nest_asyncio. "
                    "Use train_from_labeled_async() instead or install nest_asyncio."
                )
            return asyncio.run(
                cls.train_from_labeled_async(
                    data,
                    text_column,
                    label_column,
                    id_column=id_column,
                    taxonomy=taxonomy,
                    embedding_model=embedding_model,
                    test_size=test_size,
                    verbosity=verbosity,
                )
            )

    # =========================================================================
    # Binary Detection Methods
    # =========================================================================

    @classmethod
    async def find_matches_async(
        cls,
        data: Union[str, Path, pd.DataFrame, List[Doc]],
        category: Dict[str, Any],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        threshold: float = 0.5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        case_sensitive: bool = False,
        embedding_model: str = "text-embedding-3-large",
        verbosity: Verbosity = Verbosity.NORMAL,
    ) -> MatchResult:
        """Find documents matching a single category using hybrid matching.

        This is a fast, lightweight alternative to full taxonomy generation when you
        want to find documents related to ONE specific category. Uses embedding
        similarity combined with keyword matching for robust detection.

        Args:
            data: Documents to search. Can be:
                - Path to CSV/JSON file
                - pandas DataFrame
                - List of Doc objects
            category: Category definition with:
                - name (str): Category name
                - description (str): What this category represents
                - keywords (list[str], optional): Keywords to boost matching
            text_column: Column containing text (required for CSV/DataFrame)
            id_column: Optional column for document IDs
            threshold: Minimum score to consider a match (0-1, default: 0.5)
            semantic_weight: Weight for semantic similarity (default: 0.7)
            keyword_weight: Weight for keyword matching (default: 0.3)
            case_sensitive: Whether keyword matching is case-sensitive (default: False)
            embedding_model: OpenAI embedding model (default: text-embedding-3-large)
            verbosity: Output verbosity level

        Returns:
            MatchResult with matched documents, scores, and statistics

        Example:
            >>> matches = await Delve.find_matches_async(
            ...     "traces.csv",
            ...     category={
            ...         "name": "Refund Request",
            ...         "description": "User asking for refund or money back",
            ...         "keywords": ["refund", "money back", "cancel order"]
            ...     },
            ...     text_column="content",
            ...     threshold=0.6,
            ... )
            >>> print(f"Found {len(matches.documents)} matching traces")
        """
        import numpy as np
        from langchain_openai import OpenAIEmbeddings

        console = Console(verbosity)

        # Validate category
        if "name" not in category:
            raise ValueError("Category must have a 'name' field")
        if "description" not in category:
            raise ValueError("Category must have a 'description' field")

        keywords = category.get("keywords", [])
        if not keywords and keyword_weight > 0:
            console.warning(
                "No keywords provided but keyword_weight > 0. "
                "Score will be based entirely on semantic similarity."
            )
            # Adjust weights when no keywords
            semantic_weight = 1.0
            keyword_weight = 0.0

        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        semantic_weight = semantic_weight / total_weight
        keyword_weight = keyword_weight / total_weight

        # Load documents
        docs = cls._load_docs_for_classification(data, text_column, id_column)
        console.info(f"Loaded {len(docs)} documents to search")

        # Generate embeddings
        with console.status("Generating embeddings..."):
            encoder = OpenAIEmbeddings(model=embedding_model)

            # Embed category description
            category_embedding = np.array(
                await encoder.aembed_documents([category["description"]])
            )[0]

            # Embed all documents
            doc_embeddings = np.array(
                await encoder.aembed_documents([d.content for d in docs])
            )

        # Calculate semantic scores (cosine similarity)
        with console.status("Computing similarity scores..."):
            # Normalize for cosine similarity
            category_norm = category_embedding / np.linalg.norm(category_embedding)
            norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            doc_norms = doc_embeddings / norms

            # Cosine similarity
            semantic_scores = np.dot(doc_norms, category_norm)

        # Calculate keyword scores
        keyword_scores = np.zeros(len(docs))
        if keywords:
            for i, doc in enumerate(docs):
                content = doc.content if case_sensitive else doc.content.lower()
                matches = sum(
                    1 for kw in keywords
                    if (kw if case_sensitive else kw.lower()) in content
                )
                keyword_scores[i] = matches / len(keywords)

        # Combine scores
        final_scores = (
            (semantic_weight * semantic_scores) + (keyword_weight * keyword_scores)
        )

        # Label ALL documents - matches get category name, non-matches get None
        labeled_docs = []
        match_count = 0
        for i, doc in enumerate(docs):
            is_match = final_scores[i] >= threshold
            if is_match:
                match_count += 1

            labeled_doc = Doc(
                id=doc.id,
                content=doc.content,
                category=category["name"] if is_match else None,
                confidence=float(final_scores[i]),
            )
            # Attach additional scores as attributes
            labeled_doc.semantic_score = float(semantic_scores[i])
            labeled_doc.keyword_score = float(keyword_scores[i])
            labeled_docs.append(labeled_doc)

        # Sort by score descending
        labeled_docs.sort(key=lambda d: d.confidence, reverse=True)

        console.success(
            f"Found {match_count} matches out of {len(docs)} documents "
            f"(threshold: {threshold})"
        )

        return MatchResult(
            documents=labeled_docs,
            category=category,
            stats={
                "total_documents": len(docs),
                "matches": match_count,
                "match_rate": match_count / len(docs) if docs else 0,
                "threshold": threshold,
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
                "embedding_model": embedding_model,
                "avg_score": float(np.mean(final_scores)),
                "max_score": float(np.max(final_scores)),
                "min_score": float(np.min(final_scores)),
            },
        )

    @classmethod
    def find_matches(
        cls,
        data: Union[str, Path, pd.DataFrame, List[Doc]],
        category: Dict[str, Any],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        threshold: float = 0.5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        case_sensitive: bool = False,
        embedding_model: str = "text-embedding-3-large",
        verbosity: Verbosity = Verbosity.NORMAL,
    ) -> MatchResult:
        """Find documents matching a single category (sync version).

        This is a fast, lightweight alternative to full taxonomy generation when you
        want to find documents related to ONE specific category.

        See find_matches_async() for full documentation.

        Example:
            >>> matches = Delve.find_matches(
            ...     "traces.csv",
            ...     category={
            ...         "name": "Refund Request",
            ...         "description": "User asking for refund or money back",
            ...         "keywords": ["refund", "money back", "cancel order"]
            ...     },
            ...     text_column="content",
            ... )
            >>> for doc in matches.documents[:5]:
            ...     print(f"{doc.id}: {doc.confidence:.2f} - {doc.content[:50]}...")
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                cls.find_matches_async(
                    data,
                    category,
                    text_column=text_column,
                    id_column=id_column,
                    threshold=threshold,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                    case_sensitive=case_sensitive,
                    embedding_model=embedding_model,
                    verbosity=verbosity,
                )
            )
        else:
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Cannot use find_matches() in Jupyter/Colab without nest_asyncio. "
                    "Use find_matches_async() instead or install nest_asyncio."
                )
            return asyncio.run(
                cls.find_matches_async(
                    data,
                    category,
                    text_column=text_column,
                    id_column=id_column,
                    threshold=threshold,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                    case_sensitive=case_sensitive,
                    embedding_model=embedding_model,
                    verbosity=verbosity,
                )
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _load_docs_for_classification(
        data: Union[str, Path, pd.DataFrame, List[Doc]],
        text_column: Optional[str],
        id_column: Optional[str],
    ) -> List[Doc]:
        """Load documents from various sources for classification."""
        if isinstance(data, list):
            # Already Doc objects
            if data and isinstance(data[0], Doc):
                return data
            raise ValueError("List must contain Doc objects")

        # Load as DataFrame
        df = Delve._load_dataframe(data)

        if text_column is None:
            raise ValueError("text_column is required for CSV/DataFrame input")

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found. Available: {list(df.columns)}")

        docs = []
        for idx, row in df.iterrows():
            doc_id = str(row[id_column]) if id_column and id_column in df.columns else str(idx)
            docs.append(Doc(
                id=doc_id,
                content=str(row[text_column]),
            ))

        return docs

    @staticmethod
    def _load_dataframe(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data as a pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data

        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .json")
