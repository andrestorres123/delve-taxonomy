"""Pydantic schemas enforced via structured outputs on all LLM calls."""

from typing import List, Optional

from pydantic import BaseModel, Field


class TaxonomyCluster(BaseModel):
    """A single category in the taxonomy."""

    id: str = Field(
        description="Category number, starting from 1 in an incremental manner."
    )
    name: str = Field(
        description="Concise category name (verb phrase or noun phrase)."
    )
    description: str = Field(
        description="Description that differentiates this category from the others."
    )


class TaxonomyGeneration(BaseModel):
    """Generated cluster table with reasoning."""

    explanation: str = Field(
        description="Explanation of why the data was clustered this way."
    )
    clusters: List[TaxonomyCluster] = Field(
        description="The generated cluster table."
    )


class TaxonomyReview(BaseModel):
    """Review of a reference cluster table, including the updated table."""

    rating_score: int = Field(
        description="Quality rating of the reference table, integer between 0 and 100."
    )
    explanation: str = Field(description="Explanation of the rating score.")
    suggestions: str = Field(
        description="Suggested edits, or 'N/A' if no edits are needed."
    )
    clusters: List[TaxonomyCluster] = Field(
        description="The updated cluster table, or the original table if no edits were made."
    )


class DocumentSummary(BaseModel):
    """Summary of a single document."""

    summary: str = Field(description="The document summary.")
    explanation: str = Field(description="How the summary was written.")


class CategoryLabel(BaseModel):
    """Category assignment for a single document."""

    reasoning: str = Field(
        description="Chain of reasoning for why the chosen category fits the content."
    )
    category_id: Optional[str] = Field(
        default=None,
        description="Numeric ID of the chosen category, or null if no category fits.",
    )
