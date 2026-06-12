"""Node for updating taxonomies based on new document batches."""

from langchain_core.runnables import RunnableConfig

from delve.configuration import Configuration
from delve.prompts import TAXONOMY_UPDATE_PROMPT
from delve.schemas import TaxonomyReview
from delve.state import State
from delve.utils import clusters_to_dict, invoke_taxonomy_chain, load_chat_model


def _setup_update_chain(configuration: Configuration):
    """Set up the chain for taxonomy updates.

    Returns:
        Chain for updating taxonomies.
    """
    model = load_chat_model(configuration.fast_llm)

    return (
        TAXONOMY_UPDATE_PROMPT
        | model.with_structured_output(TaxonomyReview)
        | (lambda result: clusters_to_dict(result.clusters))
    ).with_config(run_name="UpdateTaxonomy")


async def update_taxonomy(
    state: State,
    config: RunnableConfig
) -> dict:
    """Update taxonomy using the next batch of documents.
    
    Args:
        state: Current application state
        config: Configuration for the run
        
    Returns:
        dict: Updated state fields with revised taxonomy
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Set up the chain
    update_chain = _setup_update_chain(configuration)

    # Determine which minibatch to use
    which_mb = len(state.clusters) % len(state.minibatches)

    # Update taxonomy using the next batch
    return await invoke_taxonomy_chain(
        update_chain,
        state,
        config,
        state.minibatches[which_mb]
    )
