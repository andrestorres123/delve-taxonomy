"""JSON and JSONL file adapter for loading documents."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, List, Optional

from jsonpath_ng import parse

from delve.adapters.base import DataSource, DataSourceConfig
from delve.state import Doc


class JSONAdapter(DataSource):
    """Adapter for loading data from JSON and JSONL files.

    Supports both standard JSON arrays and JSONL (newline-delimited JSON).
    Can extract nested fields using JSONPath expressions.
    """

    def __init__(
        self,
        file_path: str,
        text_field: str = "text",
        id_field: Optional[str] = None,
        json_path: Optional[str] = None,
        encoding: str = "utf-8",
    ):
        """Initialize JSON adapter.

        Args:
            file_path: Path to the JSON or JSONL file
            text_field: Name of the field containing text content
            id_field: Optional field name for document IDs (generates UUIDs if not provided)
            json_path: Optional JSONPath expression to extract data (e.g., "$.messages[*]")
            encoding: File encoding (default: utf-8)
        """
        config = DataSourceConfig(
            source_type="json" if file_path.endswith(".json") else "jsonl",
            source_path=file_path,
            additional_params={
                "text_field": text_field,
                "id_field": id_field,
                "json_path": json_path,
                "encoding": encoding,
            },
        )
        super().__init__(config)
        self.file_path = Path(file_path)
        self.text_field = text_field
        self.id_field = id_field
        self.json_path = json_path
        self.encoding = encoding
        self.is_jsonl = file_path.endswith(".jsonl")

    def validate(self) -> bool:
        """Validate that the JSON file exists and is valid JSON.

        Returns:
            True if valid

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid JSON
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

        # Try to parse the file
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                if self.is_jsonl:
                    # JSONL: try to parse first line
                    first_line = f.readline()
                    if first_line.strip():
                        json.loads(first_line)
                else:
                    # JSON: parse entire file
                    json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read JSON file: {e}")

        return True

    def _extract_with_jsonpath(self, data: Any) -> List[dict]:
        """Extract data using JSONPath expression.

        Args:
            data: Parsed JSON data

        Returns:
            List of dictionaries extracted via JSONPath
        """
        if not self.json_path:
            # If no JSONPath, treat data as array or wrap in array
            if isinstance(data, list):
                return data
            else:
                return [data]

        # Parse JSONPath and extract matches
        jsonpath_expr = parse(self.json_path)
        matches = jsonpath_expr.find(data)
        return [match.value for match in matches]

    def _extract_field(self, item: dict, field: str) -> Optional[str]:
        """Extract a field from a dictionary, supporting nested paths.

        Args:
            item: Dictionary to extract from
            field: Field name (supports dot notation for nested fields)

        Returns:
            Field value as string, or None if not found
        """
        if "." in field:
            # Nested field (e.g., "metadata.text")
            parts = field.split(".")
            current = item
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return str(current) if current is not None else None
        else:
            # Simple field
            return str(item.get(field)) if field in item else None

    async def load(self) -> List[Doc]:
        """Load documents from JSON/JSONL file.

        Returns:
            List of Doc objects

        Raises:
            Exception: If file cannot be read or parsed
        """
        # Validate first
        self.validate()

        # Read and parse file
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                if self.is_jsonl:
                    # JSONL: parse line by line
                    data = []
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                else:
                    # JSON: parse entire file
                    data = json.load(f)
        except Exception as e:
            raise Exception(f"Failed to read JSON file: {e}")

        # Extract items using JSONPath if provided
        items = self._extract_with_jsonpath(data)

        # Convert to Doc objects
        documents = []
        for idx, item in enumerate(items):
            # Handle both dict and non-dict items
            if not isinstance(item, dict):
                # If item is a string, use it directly as content
                if isinstance(item, str):
                    doc = Doc(id=str(uuid.uuid4()), content=item)
                    documents.append(doc)
                continue

            # Extract ID
            if self.id_field:
                doc_id = self._extract_field(item, self.id_field)
                if not doc_id:
                    doc_id = str(uuid.uuid4())
            else:
                doc_id = str(uuid.uuid4())

            # Extract text content
            content = self._extract_field(item, self.text_field)

            # Skip empty content
            if not content or content.strip() == "" or content == "None":
                continue

            doc = Doc(id=doc_id, content=content)
            documents.append(doc)

        if not documents:
            raise ValueError(f"No valid documents found in JSON file: {self.file_path}")

        return documents
