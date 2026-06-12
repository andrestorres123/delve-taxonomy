"""Node for reviewing and finalizing taxonomies."""

import random

from langchain_core.runnables import RunnableConfig

from delve.configuration import Configuration
from delve.prompts import TAXONOMY_REVIEW_PROMPT
from delve.schemas import TaxonomyReview
from delve.state import State
from delve.utils import clusters_to_dict, invoke_taxonomy_chain, load_chat_model


def _setup_review_chain(configuration: Configuration):
    """Set up the chain for taxonomy review.

    Returns:
        Chain for reviewing taxonomies.
    """
    model = load_chat_model(configuration.fast_llm)

    return (
        TAXONOMY_REVIEW_PROMPT
        | model.with_structured_output(TaxonomyReview)
        | (lambda result: clusters_to_dict(result.clusters))
    ).with_config(run_name="ReviewTaxonomy")


async def review_taxonomy(
    state: State,
    config: RunnableConfig
) -> dict:
    """Review and finalize taxonomy using a random sample of documents.
    
    Args:
        state: Current application state
        config: Configuration for the run
        
    Returns:
        dict: Updated state fields with reviewed taxonomy
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Set up the chain
    review_chain = _setup_review_chain(configuration)

    # Create random sample of documents
    batch_size = configuration.batch_size
    indices = list(range(len(state.documents)))
    random.shuffle(indices)
    sample_indices = indices[:batch_size]

    # Review taxonomy using sampled documents
    return await invoke_taxonomy_chain(
        review_chain,
        state,
        config,
        sample_indices
    )
