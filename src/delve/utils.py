"""Utility & helper functions."""

import os
from typing import Dict, List, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from delve.schemas import TaxonomyCluster
from delve.state import Doc, State


def validate_api_key() -> None:
    """Validate that ANTHROPIC_API_KEY is set.

    Raises:
        ValueError: If API key is not set, with a helpful error message.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set.\n\n"
            "Please set your Anthropic API key:\n"
            "  export ANTHROPIC_API_KEY=your-api-key-here\n\n"
            "You can get an API key from: https://console.anthropic.com/\n"
            "For more information, see: https://docs.anthropic.com/claude/docs/getting-access-to-claude"
        )

    # Basic validation - check if it looks like an Anthropic key
    if not api_key.startswith("sk-ant-"):
        raise ValueError(
            "ANTHROPIC_API_KEY appears to be invalid.\n\n"
            "Anthropic API keys should start with 'sk-ant-'.\n"
            "Please check your API key and try again.\n\n"
            "You can get a new API key from: https://console.anthropic.com/"
        )


def validate_openai_api_key() -> None:
    """Validate that OPENAI_API_KEY is set (needed for embeddings/classifier).

    Raises:
        ValueError: If API key is not set, with a helpful error message.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set.\n\n"
            "This key is required for generating embeddings when using the classifier\n"
            "(when sample_size < total documents).\n\n"
            "Please set your OpenAI API key:\n"
            "  export OPENAI_API_KEY=your-api-key-here\n\n"
            "You can get an API key from: https://platform.openai.com/api-keys"
        )

    # Basic validation - check if it looks like an OpenAI key
    if not (api_key.startswith("sk-") or api_key.startswith("sess-")):
        raise ValueError(
            "OPENAI_API_KEY appears to be invalid.\n\n"
            "OpenAI API keys typically start with 'sk-'.\n"
            "Please check your API key and try again.\n\n"
            "You can get a new API key from: https://platform.openai.com/api-keys"
        )


def validate_all_api_keys(needs_openai: bool = True) -> None:
    """Validate all required API keys upfront.

    Args:
        needs_openai: Whether OpenAI key is needed (for embeddings/classifier).
                     Set to False if sample_size=0 (all docs labeled by LLM).

    Raises:
        ValueError: If any required API key is missing or invalid.
    """
    errors = []

    # Check Anthropic key (always required)
    try:
        validate_api_key()
    except ValueError as e:
        errors.append(str(e))

    # Check OpenAI key (needed for classifier/embeddings)
    if needs_openai:
        try:
            validate_openai_api_key()
        except ValueError as e:
            errors.append(str(e))

    if errors:
        separator = "\n" + "=" * 50 + "\n"
        raise ValueError(separator.join(errors))


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        
    Returns:
        BaseChatModel: The loaded chat model.
        
    Raises:
        ValueError: If API key is not configured or if model name is invalid.
    """
    try:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    except ValueError:
        raise ValueError(
            f"Invalid model name format: '{fully_specified_name}'\n"
            f"Expected format: 'provider/model-name' (e.g., 'anthropic/claude-opus-4-8')"
        )
    
    try:
        return init_chat_model(model, model_provider=provider)
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            # Re-validate and provide better error message
            validate_api_key()
            raise ValueError(
                f"Failed to authenticate with {provider}.\n\n"
                f"Error: {error_msg}\n\n"
                f"Please verify your ANTHROPIC_API_KEY is correct:\n"
                f"  export ANTHROPIC_API_KEY=your-api-key-here\n\n"
                f"You can get an API key from: https://console.anthropic.com/"
            ) from e
        else:
            raise ValueError(
                f"Failed to load model '{fully_specified_name}': {error_msg}\n\n"
                f"Please check:\n"
                f"  1. The model name is correct\n"
                f"  2. You have access to the model\n"
                f"  3. All required dependencies are installed"
            ) from e


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


# Taxonomy generation configuration defaults
TAXONOMY_CONFIG = {
    "suggestion_length": 30,
    "cluster_name_length": 10,
    "cluster_description_length": 30,
    "explanation_length": 20,
    "max_num_clusters": 5,
}


def clusters_to_dict(clusters: List[TaxonomyCluster]) -> Dict[str, List[Dict[str, str]]]:
    """Convert structured-output clusters to the dict shape used in state."""
    return {"clusters": [cluster.model_dump() for cluster in clusters]}


def format_docs(docs: List[Doc]) -> str:
    """Format documents as XML for taxonomy generation.
    
    Args:
        docs: List of documents to format
        
    Returns:
        str: XML formatted document summaries
    """
    xml_table = "<conversations>\n"
    for doc in docs:
        doc_id = doc["id"] if isinstance(doc, dict) else doc.id
        doc_summary = doc.get("summary", "") if isinstance(doc, dict) else (doc.summary or "")
        xml_table += f'<conv_summ id={doc_id}>{doc_summary}</conv_summ>\n'
    xml_table += "</conversations>"
    return xml_table


def format_taxonomy(clusters: List[Dict[str, str]]) -> str:
    """Format taxonomy clusters as XML.
    
    Args:
        clusters: List of cluster dictionaries
        
    Returns:
        str: XML formatted taxonomy
    """
    xml = "<cluster_table>\n"
    for label in clusters:
        xml += "  <cluster>\n"
        xml += f'    <id>{label["id"]}</id>\n'
        xml += f'    <name>{label["name"]}</name>\n'
        xml += f'    <description>{label["description"]}</description>\n'
        xml += "  </cluster>\n"
    xml += "</cluster_table>"
    return xml


async def invoke_taxonomy_chain(
    chain: Runnable,
    state: State,
    config: RunnableConfig,
    mb_indices: List[int],
) -> Dict[str, List[List[Dict[str, str]]]]:
    """Invoke the taxonomy generation chain."""
    try:
        configurable = config["configurable"]
        minibatch = [state.documents[idx] for idx in mb_indices]
        data_table_xml = format_docs(minibatch)

        previous_taxonomy = state.clusters[-1] if state.clusters else []
        cluster_table_xml = format_taxonomy(previous_taxonomy)

        # Format feedback (non-interactive mode, no user feedback)
        feedback = "No previous feedback provided."

        updated_taxonomy = await chain.ainvoke(
            {
                "data_xml": data_table_xml,
                "use_case": state.use_case,
                "cluster_table_xml": cluster_table_xml,
                "feedback": feedback,  # Add feedback to the prompt variables
                "suggestion_length": configurable.get(
                    "suggestion_length", 
                    TAXONOMY_CONFIG["suggestion_length"]
                ),
                "cluster_name_length": configurable.get(
                    "cluster_name_length", 
                    TAXONOMY_CONFIG["cluster_name_length"]
                ),
                "cluster_description_length": configurable.get(
                    "cluster_description_length", 
                    TAXONOMY_CONFIG["cluster_description_length"]
                ),
                "explanation_length": configurable.get(
                    "explanation_length", 
                    TAXONOMY_CONFIG["explanation_length"]
                ),
                "max_num_clusters": configurable.get(
                    "max_num_clusters", 
                    TAXONOMY_CONFIG["max_num_clusters"]
                ),
            }
        )
        return {
            "clusters": [updated_taxonomy["clusters"]],
            "status": ["Taxonomy generated.."],
        }
    except Exception:
        # Re-raise to be handled by higher-level exception handlers
        # which have access to console for proper error display
        raise
