"""Define the taxonomy generation graph structure."""

from langgraph.graph import END, START, StateGraph

from delve.configuration import Configuration
from delve.core.batch_generator import generate_minibatches
from delve.core.data_loader import load_data
from delve.core.document_labeler import label_documents
from delve.core.results_saver import save_results
from delve.core.summarizer import generate_summaries
from delve.core.taxonomy_generator import generate_taxonomy
from delve.core.taxonomy_reviewer import review_taxonomy
from delve.core.taxonomy_updater import update_taxonomy
from delve.routing import should_discover_taxonomy, should_review
from delve.state import InputState, State

# Create the graph
builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

# Add nodes
builder.add_node("load_data", load_data)
builder.add_node("summarize", generate_summaries)
builder.add_node("get_minibatches", generate_minibatches)
builder.add_node("generate_taxonomy", generate_taxonomy)
builder.add_node("update_taxonomy", update_taxonomy)
builder.add_node("review_taxonomy", review_taxonomy)
builder.add_node("label_documents", label_documents)
builder.add_node("save_results", save_results)

# Add edges
builder.add_edge(START, "load_data")

# Conditional routing: skip discovery if predefined taxonomy is loaded
builder.add_conditional_edges(
    "load_data",
    should_discover_taxonomy,
    {
        "summarize": "summarize",  # Start discovery flow
        "label_documents": "label_documents",  # Skip to labeling
    },
)

# Discovery flow edges
builder.add_edge("summarize", "get_minibatches")
builder.add_edge("get_minibatches", "generate_taxonomy")
builder.add_edge("generate_taxonomy", "update_taxonomy")

# Review and labeling edges
builder.add_edge("review_taxonomy", "label_documents")
builder.add_edge("label_documents", "save_results")
builder.add_edge("save_results", END)

# Add conditional edges for the review process
builder.add_conditional_edges(
    "update_taxonomy",
    should_review,
    {
        "update_taxonomy": "update_taxonomy",
        "review_taxonomy": "review_taxonomy",
    },
)

# Compile the graph (no interrupts - fully automated)
graph = builder.compile()
graph.name = "Delve Taxonomy Generator"
