"""Base classes for data source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from delve.state import Doc


@dataclass
class DataSourceConfig:
    """Configuration for a data source.

    Attributes:
        source_type: Type of data source (csv, json, langsmith, dataframe)
        source_path: Optional path to the data source file
        additional_params: Additional adapter-specific parameters
    """

    source_type: str
    source_path: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for data source adapters.

    All data source adapters must implement this interface to provide
    a consistent way to load documents from different sources.
    """

    def __init__(self, config: DataSourceConfig):
        """Initialize the data source adapter.

        Args:
            config: Configuration for this data source
        """
        self.config = config

    @abstractmethod
    async def load(self) -> List[Doc]:
        """Load documents from the data source.

        Returns:
            List of Doc objects with id and content fields populated

        Raises:
            Exception: If the data source cannot be accessed or parsed
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the data source is accessible and properly configured.

        Returns:
            True if valid

        Raises:
            ValueError: If the configuration is invalid
            FileNotFoundError: If the data source cannot be found
            Exception: For other validation errors
        """
        pass
