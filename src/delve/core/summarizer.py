"""Node for generating summaries of documents."""

from typing import Any, Dict, List
from uuid import uuid4

from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough

from delve.configuration import Configuration
from delve.prompts import SUMMARY_PROMPT
from delve.schemas import DocumentSummary
from delve.state import State
from delve.utils import load_chat_model


def _normalize_doc(doc: Any) -> Dict[str, str]:
    """Normalize one document to a ``{"id", "content"}`` dict for summarization.

    Handles the three shapes that can reach the summarizer:
      * ``str`` — raw text, gets a fresh id;
      * ``dict`` — used as-is, gets a fresh id if missing;
      * ``Doc`` (or any object with a ``.content`` attribute) — fields are read
        off the object.

    The last case is the common one: ``state.documents`` is a ``List[Doc]``.
    Stringifying a ``Doc`` (``str(doc)``) would poison ``content`` with the full
    ``Doc(...)`` repr, which then flows into summaries, classifier embeddings, and
    CSV exports. Pulling ``.content`` off the object avoids that.
    """
    if isinstance(doc, str):
        return {"id": str(uuid4()), "content": doc}
    if isinstance(doc, dict):
        return {"id": doc.get("id") or str(uuid4()), "content": doc.get("content", "")}
    if hasattr(doc, "content"):
        return {"id": getattr(doc, "id", None) or str(uuid4()), "content": doc.content}
    return {"id": str(uuid4()), "content": str(doc)}


def _get_content(state: Dict[str, List]) -> List[Dict[str, str]]:
    """Extract content from documents for summarization.

    Args:
        state: State dictionary containing documents

    Returns:
        List of document contents formatted for summarization
    """
    docs = state["documents"]
    return [
        {
            "content": (
                doc["content"] if isinstance(doc, dict)
                else doc.content
            )
        }
        for doc in docs
    ]


def _summary_dict(result: Any) -> dict:
    """Convert the structured summary output to the dict shape used downstream."""
    return {"summary": result.summary, "explanation": result.explanation}


def _reduce_summaries(combined: Dict[str, Any]) -> Dict[str, Any]:
    """Combine documents with their summaries.

    Args:
        combined: Dictionary containing documents and their summaries

    Returns:
        dict: Documents enriched with summaries and explanations
    """
    summaries = combined["summaries"]
    documents = combined["documents"]
    return {
        "documents": [
            {
                "id": doc.get("id", str(uuid4())),
                "content": doc.get("content", ""),
                "summary": summ_info.get("summary", ""),
                "explanation": summ_info.get("explanation", ""),
            }
            for doc, summ_info in zip(documents, summaries)
        ],
        "status": ["Summarized successfully."],
    }


async def generate_summaries(
    state: State,
    config: RunnableConfig,
) -> dict:
    """Generate summaries for a collection of documents."""
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model and prompt
    try:
        model = load_chat_model(configuration.fast_llm)
    except (ValueError, TypeError) as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            raise ValueError(
                f"Authentication failed when loading model '{configuration.fast_llm}'.\n\n"
                f"{error_msg}\n\n"
                f"Please verify your ANTHROPIC_API_KEY is set correctly:\n"
                f"  export ANTHROPIC_API_KEY=your-api-key-here\n\n"
                f"You can get an API key from: https://console.anthropic.com/"
            ) from e
        raise

    summary_prompt = SUMMARY_PROMPT.partial(
        summary_length=20, explanation_length=30
    )

    # Create the summary chain
    summary_chain = (
        summary_prompt
        | model.with_structured_output(DocumentSummary)
        | RunnableLambda(_summary_dict)
    ).with_config(run_name="GenerateSummary")

    # Create the full chain with map-reduce
    map_reduce_chain = (
        RunnablePassthrough.assign(
            summaries=_get_content
            | RunnableLambda(func=summary_chain.batch, afunc=summary_chain.abatch)
        )
        | _reduce_summaries
    )

    # Process documents
    processed_docs = [_normalize_doc(doc) for doc in state.documents]

    try:
        return await map_reduce_chain.ainvoke({"documents": processed_docs})
    except (TypeError, ValueError) as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower() or "could not resolve" in error_msg.lower():
            raise ValueError(
                f"Authentication failed during document summarization.\n\n"
                f"The API key is missing or invalid. Please verify:\n"
                f"  export ANTHROPIC_API_KEY=your-api-key-here\n\n"
                f"Get your API key from: https://console.anthropic.com/\n\n"
                f"Original error: {error_msg}"
            ) from e
        raise
