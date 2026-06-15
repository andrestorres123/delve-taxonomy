"""LangSmith adapter for loading documents from LangSmith projects."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from langsmith import Client
from langsmith.schemas import Run

from delve.adapters.base import DataSource, DataSourceConfig
from delve.state import Doc


def to_xml(
    data: Union[Dict, List],
    tag_name: str,
    *,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    nested: Optional[List[str]] = None,
    body_key: Optional[str] = None,
    list_item_tag: str = "item",
    max_body_length: Optional[int] = None,
) -> str:
    """Convert data structure to XML format.

    Args:
        data: The data to convert
        tag_name: The name of the root tag
        exclude: Keys to exclude from the output
        include: Keys to include in the output (if None, include all)
        nested: Keys that should be processed as nested structures
        body_key: Key whose value should be used as the tag body
        list_item_tag: Tag name to use for list items
        max_body_length: Maximum length for body text before truncating

    Returns:
        str: The XML representation of the data
    """
    skip = exclude or []
    nested = nested or []

    def process_dict(d: Dict) -> tuple[str, str]:
        attr_str = ""
        body = ""

        for key, value in d.items():
            if key == body_key:
                body += str(value)
                continue
            if value is None or key in skip:
                continue
            if include and key not in include:
                continue
            if key in nested:
                body += process_value(value, key)
            elif isinstance(value, (dict, list)):
                body += f"<{key}>{process_value(value, key)}</{key}>"
            else:
                attr_str += f' {key}="{value}"'

        return attr_str, body

    def process_value(value: Union[Dict, List, str, int, float], key: str) -> str:
        if isinstance(value, dict):
            attr, body = process_dict(value)

            if max_body_length and len(body) > max_body_length:
                body = body[:max_body_length] + "..."
            return f"<{key}{attr}>{body}</{key}>"
        elif isinstance(value, list):
            res = "".join(
                f"<{list_item_tag}>{process_value(item, list_item_tag)}</{list_item_tag}>"
                for item in value
            )
            if max_body_length and len(res) > max_body_length:
                res = res[:max_body_length] + "..."
            return res
        else:
            val = str(value)
            if max_body_length and len(val) > max_body_length:
                val = val[:max_body_length] + "..."
            return val

    if isinstance(data, dict):
        attr_str, body = process_dict(data)
        return f"<{tag_name}{attr_str}>{body}</{tag_name}>"
    elif isinstance(data, (list, tuple)):
        body = "".join(
            f"<{list_item_tag}>{process_value(item, list_item_tag)}</{list_item_tag}>"
            for item in data
        )
        return f"<{tag_name}>{body}</{tag_name}>"
    else:
        raise ValueError("Input must be a dictionary or a list")


def run_to_doc(run: Run, max_length: int = 500) -> Doc:
    """Convert a LangSmith run to a document.

    Args:
        run: The LangSmith run to convert
        max_length: Maximum length for content fields

    Returns:
        Doc: A document containing the run's content
    """
    inputs_str = to_xml(
        run.inputs,
        "inputs",
        include=["messages", "content", "type", "chat_history"],
        exclude=["__end__", "id"],
        max_body_length=max_length,
        body_key="content",
    )
    outputs_str = ""
    if run.outputs:
        outputs_str = "\n" + to_xml(
            run.outputs,
            "outputs",
            include=["answer"],
            exclude=["__end__", "documents"],
            max_body_length=max_length,
            body_key="answer",
        )
    return Doc(
        id=str(run.id),
        content=f"{inputs_str}{outputs_str}",
    )


class LangSmithAdapter(DataSource):
    """Adapter for loading data from LangSmith projects.

    Retrieves runs from a LangSmith project and converts them to Doc objects.
    """

    def __init__(
        self,
        project_name: str,
        api_key: str,
        days: int = 7,
        max_runs: int = 500,
        sample_size: Optional[int] = None,
        filter_expr: str = "eq(is_root, true)",
    ):
        """Initialize LangSmith adapter.

        Args:
            project_name: Name of the LangSmith project
            api_key: LangSmith API key
            days: Number of days to look back for runs
            max_runs: Maximum number of runs to retrieve
            sample_size: Optional number of runs to sample (if None, use all)
            filter_expr: Filter expression for runs (default: root runs only)
        """
        config = DataSourceConfig(
            source_type="langsmith",
            source_path=f"langsmith://{project_name}",
            additional_params={
                "project_name": project_name,
                "api_key": api_key,
                "days": days,
                "max_runs": max_runs,
                "sample_size": sample_size,
                "filter_expr": filter_expr,
            },
        )
        super().__init__(config)
        self.project_name = project_name
        self.api_key = api_key
        self.days = days
        self.max_runs = max_runs
        self.sample_size = sample_size
        self.filter_expr = filter_expr
        self.client: Optional[Client] = None

    def validate(self) -> bool:
        """Validate LangSmith connection and project access.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
            Exception: If cannot connect to LangSmith
        """
        if not self.project_name:
            raise ValueError("project_name is required")

        if not self.api_key:
            raise ValueError("api_key is required")

        # Try to create client and test connection
        try:
            self.client = Client(api_key=self.api_key)
            # Test by trying to list projects (minimal API call)
            list(self.client.list_projects(limit=1))
        except Exception as e:
            raise Exception(f"Failed to connect to LangSmith: {e}")

        return True

    async def load(self) -> List[Doc]:
        """Load documents from LangSmith project.

        Returns:
            List of Doc objects

        Raises:
            Exception: If runs cannot be retrieved
        """
        # Validate first
        self.validate()
        assert self.client is not None  # validate() initializes the client

        # Calculate lookback time
        delta_days = datetime.now() - timedelta(days=self.days)

        # Retrieve runs
        try:
            runs = list(
                self.client.list_runs(
                    project_name=self.project_name,
                    filter=self.filter_expr,
                    start_time=delta_days,
                    select=["inputs", "outputs"],
                    limit=self.max_runs,
                )
            )
        except Exception as e:
            raise Exception(f"Failed to retrieve runs from LangSmith: {e}")

        if not runs:
            raise ValueError(
                f"No runs found in LangSmith project '{self.project_name}' "
                f"for the last {self.days} days"
            )

        # Convert runs to documents
        documents = [run_to_doc(run) for run in runs]

        # Apply sampling if specified
        if self.sample_size is not None and self.sample_size < len(documents):
            documents = random.sample(documents, self.sample_size)

        return documents
