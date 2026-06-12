"""Node for loading data using adapters."""

import json
import random
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from langchain_core.runnables import RunnableConfig

from delve.configuration import Configuration
from delve.state import State


def _load_predefined_taxonomy(taxonomy_input: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Load taxonomy from file or dict.

    Args:
        taxonomy_input: Either a list of taxonomy dicts or a file path (JSON/CSV)

    Returns:
        List of taxonomy dictionaries with 'id', 'name', 'description'

    Raises:
        ValueError: If taxonomy format is invalid or file cannot be read
    """
    if isinstance(taxonomy_input, list):
        # Already in correct format - validate it has required fields
        for item in taxonomy_input:
            if not all(k in item for k in ['id', 'name', 'description']):
                raise ValueError(
                    "Each taxonomy item must have 'id', 'name', and 'description' fields. "
                    f"Got: {item.keys()}"
                )
        return taxonomy_input

    if isinstance(taxonomy_input, str):
        # Load from file
        path = Path(taxonomy_input)

        if not path.exists():
            raise ValueError(f"Taxonomy file not found: {path}")

        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
                # Handle both direct list and nested structure
                if isinstance(data, dict) and 'taxonomy' in data:
                    data = data['taxonomy']
                if isinstance(data, dict) and 'clusters' in data:
                    data = data['clusters']
                return data

        elif path.suffix == '.csv':
            df = pd.read_csv(path)
            required_cols = {'id', 'name', 'description'}
            if not required_cols.issubset(df.columns):
                raise ValueError(
                    f"CSV must have columns: {required_cols}. "
                    f"Got: {set(df.columns)}"
                )
            return [
                {
                    "id": str(row['id']),
                    "name": str(row['name']),
                    "description": str(row['description'])
                }
                for _, row in df.iterrows()
            ]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .csv")

    raise ValueError(
        f"Invalid taxonomy format. Expected list of dicts or file path, got {type(taxonomy_input)}"
    )


async def load_data(state: State, config: RunnableConfig) -> dict:
    """Load documents using the configured adapter.

    This node expects that documents have already been loaded by the adapter
    and passed into the state. It performs sampling if configured.

    Args:
        state: Current application state with documents
        config: Configuration for the run

    Returns:
        dict: Updated state fields with sampled documents
    """
    configuration = Configuration.from_runnable_config(config)

    # Documents should already be in state.all_documents
    # (loaded by SDK/CLI before invoking the graph)
    all_docs = state.all_documents

    if not all_docs:
        raise ValueError("No documents found in state. Documents should be loaded before invoking the graph.")

    # Sample documents if sample_size is configured
    sample_size = configuration.sample_size

    if sample_size and sample_size < len(all_docs):
        # Random sample
        sampled_docs = random.sample(all_docs, sample_size)
        status_message = f"Sampled {sample_size} documents from {len(all_docs)} total documents"
    else:
        # Use all documents
        sampled_docs = all_docs
        status_message = f"Processing all {len(all_docs)} documents"

    # Check for predefined taxonomy
    result = {
        "documents": sampled_docs,
        "use_case": configuration.use_case,  # Pass use_case to state
        "status": [status_message],
    }

    if configuration.predefined_taxonomy:
        try:
            taxonomy = _load_predefined_taxonomy(configuration.predefined_taxonomy)
            result["clusters"] = [taxonomy]  # Format as List[List[Dict]] to match state structure
            result["status"].append(f"Using predefined taxonomy with {len(taxonomy)} categories")
        except Exception as e:
            raise ValueError(f"Failed to load predefined taxonomy: {e}")

    return result
