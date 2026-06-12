"""Data source adapters for loading documents from various sources."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from delve.adapters.base import DataSource, DataSourceConfig
from delve.adapters.csv_adapter import CSVAdapter
from delve.adapters.dataframe_adapter import DataFrameAdapter
from delve.adapters.json_adapter import JSONAdapter
from delve.adapters.langsmith_adapter import LangSmithAdapter


def create_adapter(
    source: Union[str, Path, pd.DataFrame],
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    source_type: Optional[str] = None,
    **kwargs,
) -> DataSource:
    """Create the appropriate adapter for a data source.

    Automatically detects the source type based on:
    - File extension (.csv, .json, .jsonl)
    - URI scheme (langsmith://)
    - Object type (pandas DataFrame)
    - Explicit source_type parameter

    Args:
        source: Data source (file path, URI, or DataFrame)
        text_column: Column/field containing text content
        id_column: Optional column/field for document IDs
        source_type: Force specific adapter type (csv, json, jsonl, langsmith, dataframe, auto)
        **kwargs: Additional adapter-specific parameters

    Returns:
        DataSource: Appropriate adapter instance

    Raises:
        ValueError: If source type cannot be determined or is not supported

    Examples:
        >>> # CSV file
        >>> adapter = create_adapter("data.csv", text_column="text")

        >>> # JSON with JSONPath
        >>> adapter = create_adapter(
        ...     "data.json",
        ...     text_field="content",
        ...     json_path="$.messages[*]"
        ... )

        >>> # LangSmith
        >>> adapter = create_adapter(
        ...     "langsmith://my-project",
        ...     api_key="lsv2_...",
        ...     days=7
        ... )

        >>> # DataFrame
        >>> adapter = create_adapter(df, text_column="text")
    """
    # Auto-detect if not specified
    if source_type is None or source_type == "auto":
        source_type = _detect_source_type(source)

    # Create appropriate adapter
    if source_type == "csv":
        if not text_column:
            raise ValueError("text_column is required for CSV adapter")
        return CSVAdapter(
            file_path=str(source),
            text_column=text_column,
            id_column=id_column,
            **kwargs,
        )

    elif source_type in ("json", "jsonl"):
        text_field = kwargs.pop("text_field", text_column or "text")
        json_path = kwargs.pop("json_path", None)
        return JSONAdapter(
            file_path=str(source),
            text_field=text_field,
            id_field=id_column,
            json_path=json_path,
            **kwargs,
        )

    elif source_type == "langsmith":
        # Parse langsmith:// URI
        if isinstance(source, str) and source.startswith("langsmith://"):
            project_name = source.replace("langsmith://", "")
        else:
            project_name = kwargs.pop("project_name", str(source))

        api_key = kwargs.pop("api_key", None)
        if not api_key:
            raise ValueError("api_key is required for LangSmith adapter")

        days = kwargs.pop("days", 7)
        max_runs = kwargs.pop("max_runs", 500)
        sample_size = kwargs.pop("sample_size", None)
        filter_expr = kwargs.pop("filter_expr", "eq(is_root, true)")

        return LangSmithAdapter(
            project_name=project_name,
            api_key=api_key,
            days=days,
            max_runs=max_runs,
            sample_size=sample_size,
            filter_expr=filter_expr,
        )

    elif source_type == "dataframe":
        if not isinstance(source, pd.DataFrame):
            raise ValueError("Source must be a pandas DataFrame for dataframe adapter")
        if not text_column:
            raise ValueError("text_column is required for DataFrame adapter")
        return DataFrameAdapter(
            df=source,
            text_column=text_column,
            id_column=id_column,
        )

    else:
        raise ValueError(
            f"Unsupported source type: {source_type}. "
            f"Supported types: csv, json, jsonl, langsmith, dataframe"
        )


def _detect_source_type(source: Union[str, Path, pd.DataFrame]) -> str:
    """Detect the source type from the source object.

    Args:
        source: Data source to detect

    Returns:
        str: Detected source type

    Raises:
        ValueError: If source type cannot be determined
    """
    # DataFrame
    if isinstance(source, pd.DataFrame):
        return "dataframe"

    # Convert to string for further detection
    source_str = str(source)

    # LangSmith URI
    if source_str.startswith("langsmith://"):
        return "langsmith"

    # File extensions
    if source_str.endswith(".csv"):
        return "csv"
    elif source_str.endswith(".json"):
        return "json"
    elif source_str.endswith(".jsonl"):
        return "jsonl"

    # Could not determine
    raise ValueError(
        f"Cannot auto-detect source type for: {source}. "
        f"Please specify source_type explicitly (csv, json, jsonl, langsmith, dataframe)"
    )


__all__ = [
    "DataSource",
    "DataSourceConfig",
    "CSVAdapter",
    "JSONAdapter",
    "LangSmithAdapter",
    "DataFrameAdapter",
    "create_adapter",
]
