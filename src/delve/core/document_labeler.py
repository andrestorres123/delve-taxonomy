"""Node for labeling documents using the generated taxonomy.

Uses a hybrid approach:
1. LLM labels sampled documents (training set)
2. If there are more documents, trains a classifier and uses it for the rest
3. Returns all labeled documents
"""

import re
from collections import Counter
from typing import Dict, Any, List, Optional
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings

from delve.state import State, Doc
from delve.utils import load_chat_model
from delve.configuration import Configuration
from delve.prompts import LABELER_PROMPT
from delve.core.classifier import train_classifier, predict_with_classifier, get_prediction_confidence


def _get_category_name_by_id(category_id: str, taxonomy: List[Dict[str, str]]) -> str:
    """Map category ID to category name.

    Args:
        category_id: The category ID as string
        taxonomy: List of category dicts with 'id', 'name', 'description'

    Returns:
        Category name

    Raises:
        ValueError: If ID not found in taxonomy
    """
    for cat in taxonomy:
        if str(cat["id"]) == str(category_id):
            return cat["name"]

    # ID not found
    available_ids = [cat["id"] for cat in taxonomy]
    raise ValueError(
        f"Category ID '{category_id}' not found in taxonomy. "
        f"Available IDs: {available_ids}"
    )


def _parse_labels(output_text: str, console=None) -> Dict[str, str]:
    """Parse the generated category ID from LLM output."""
    # Extract category ID from <category_id>N</category_id> tags
    id_matches = re.findall(
        r"<category_id>\s*(\d+)\s*</category_id>",
        output_text,
        re.DOTALL,
    )

    if not id_matches:
        # Fallback: try to find any number in the output
        if console:
            console.warning(f"No <category_id> tag found in output: {output_text[:200]}")
        return {"category_id": None}

    if len(id_matches) > 1:
        if console:
            console.warning(f"Multiple category IDs found: {id_matches}, using first one")

    return {"category_id": id_matches[0]}


def _format_taxonomy(clusters: List[Dict[str, str]]) -> str:
    """Format taxonomy clusters as XML."""

    xml = "<cluster_table>\n"

    if clusters and isinstance(clusters[0], list):
        clusters = clusters[0]

    if isinstance(clusters, dict):
        clusters = [clusters]

    for cluster in clusters:
        xml += "  <cluster>\n"
        if isinstance(cluster, dict):
            xml += f'    <id>{cluster["id"]}</id>\n'
            xml += f'    <name>{cluster["name"]}</name>\n'
            xml += f'    <description>{cluster["description"]}</description>\n'
        else:
            xml += f'    <id>{getattr(cluster, "id", "")}</id>\n'
            xml += f'    <name>{getattr(cluster, "name", "")}</name>\n'
            xml += f'    <description>{getattr(cluster, "description", "")}</description>\n'
        xml += "  </cluster>\n"
    xml += "</cluster_table>"
    return xml


def _setup_classification_chain(configuration: Configuration):
    """Set up the chain for document labeling."""
    model = load_chat_model(configuration.fast_llm)

    return (
        LABELER_PROMPT
        | model
        | StrOutputParser()
        | _parse_labels
    ).with_config(run_name="LabelDocs")


def _find_similar_documents(
    query_embeddings: np.ndarray,
    pool_embeddings: np.ndarray,
    k: int,
    exclude_indices: set = None,
) -> List[int]:
    """Find k most similar documents using cosine similarity.

    Args:
        query_embeddings: Embeddings to find similar documents for (can be multiple)
        pool_embeddings: Pool of document embeddings to search
        k: Number of similar documents to return
        exclude_indices: Indices to exclude from results

    Returns:
        List of indices into pool_embeddings
    """
    if exclude_indices is None:
        exclude_indices = set()

    # Normalize for cosine similarity
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    pool_norm = pool_embeddings / np.linalg.norm(pool_embeddings, axis=1, keepdims=True)

    # Compute similarities (average if multiple query embeddings)
    if len(query_norm.shape) == 1:
        query_norm = query_norm.reshape(1, -1)

    similarities = np.mean(np.dot(pool_norm, query_norm.T), axis=1)

    # Sort by similarity (descending) and filter excluded indices
    sorted_indices = np.argsort(similarities)[::-1]
    result = []
    for idx in sorted_indices:
        if idx not in exclude_indices:
            result.append(int(idx))
            if len(result) >= k:
                break

    return result


async def _balance_sample(
    llm_labeled_docs: List[Doc],
    unlabeled_pool: List,
    taxonomy: List[Dict[str, str]],
    min_per_cat: int,
    labeling_chain,
    encoder,
    console,
) -> tuple[List[Doc], int]:
    """Augment the training sample by finding more examples of underrepresented categories.

    Args:
        llm_labeled_docs: Documents already labeled by LLM
        unlabeled_pool: Pool of unlabeled documents to search
        taxonomy: List of category dicts
        min_per_cat: Minimum examples required per category
        labeling_chain: Chain for LLM labeling
        encoder: Embeddings encoder
        console: Console for output

    Returns:
        Tuple of (augmented docs list, count of augmented docs)
    """
    # Count current distribution
    sample_dist = Counter(doc.category for doc in llm_labeled_docs if doc.category != "Other")
    all_category_names = [cat["name"] for cat in taxonomy]

    # Find underrepresented categories
    underrepresented = []
    for cat_name in all_category_names:
        current_count = sample_dist.get(cat_name, 0)
        if current_count < min_per_cat:
            underrepresented.append({
                "name": cat_name,
                "current": current_count,
                "needed": min_per_cat - current_count,
            })

    if not underrepresented:
        console.debug("All categories have sufficient examples")
        return llm_labeled_docs, 0

    console.info(f"Found {len(underrepresented)} underrepresented categories:")
    for cat in underrepresented:
        console.info(f"  {cat['name']}: {cat['current']}/{min_per_cat} (need {cat['needed']} more)")

    if not unlabeled_pool:
        console.warning("No unlabeled documents available for balancing")
        return llm_labeled_docs, 0

    console.info(f"Searching {len(unlabeled_pool)} unlabeled documents for similar examples...")

    # Generate embeddings for unlabeled pool
    with console.status("Generating embeddings for candidate pool..."):
        pool_contents = [
            doc["content"] if isinstance(doc, dict) else doc.content
            for doc in unlabeled_pool
        ]
        pool_embeddings = np.array(await encoder.aembed_documents(pool_contents))

    augmented_docs = []
    used_indices = set()
    taxonomy_xml = _format_taxonomy(taxonomy)

    for cat_info in underrepresented:
        cat_name = cat_info["name"]
        needed = cat_info["needed"]

        # Get exemplars for this category
        exemplars = [d for d in llm_labeled_docs if d.category == cat_name]
        cat_dict = next((c for c in taxonomy if c["name"] == cat_name), None)

        if exemplars:
            # Find documents similar to existing exemplars
            with console.status(f"Finding similar docs for '{cat_name}'..."):
                exemplar_contents = [d.content for d in exemplars]
                exemplar_embeddings = np.array(await encoder.aembed_documents(exemplar_contents))

            similar_indices = _find_similar_documents(
                exemplar_embeddings,
                pool_embeddings,
                k=needed * 2,
                exclude_indices=used_indices,
            )
        elif cat_dict:
            # No exemplars - find documents similar to category description
            with console.status(f"Finding docs matching description for '{cat_name}'..."):
                desc_embedding = np.array(await encoder.aembed_documents([cat_dict["description"]]))

            similar_indices = _find_similar_documents(
                desc_embedding,
                pool_embeddings,
                k=needed * 2,
                exclude_indices=used_indices,
            )
        else:
            console.warning(f"No exemplars or description found for category '{cat_name}'")
            continue

        if not similar_indices:
            console.warning(f"No similar documents found for category '{cat_name}'")
            continue

        # Label candidates with LLM and filter those that match
        added_for_cat = 0
        for idx in similar_indices:
            if added_for_cat >= needed:
                break

            doc = unlabeled_pool[idx]
            content = doc["content"] if isinstance(doc, dict) else doc.content
            doc_id = doc["id"] if isinstance(doc, dict) else doc.id

            # Label with LLM
            result = await labeling_chain.ainvoke({
                "content": content,
                "taxonomy": taxonomy_xml,
            })

            category_id = result.get("category_id")
            if category_id:
                try:
                    assigned_cat = _get_category_name_by_id(category_id, taxonomy)
                    if assigned_cat == cat_name:
                        # Match! Add to augmented docs
                        augmented_docs.append(Doc(
                            id=doc_id,
                            content=content,
                            summary=doc.get("summary", "") if isinstance(doc, dict) else (getattr(doc, "summary", None) or ""),
                            explanation=None,
                            category=assigned_cat,
                        ))
                        used_indices.add(idx)
                        added_for_cat += 1
                except ValueError:
                    pass  # Skip invalid category IDs

        console.info(f"  Added {added_for_cat} examples for '{cat_name}'")

    if not augmented_docs:
        console.info("No additional examples found that match underrepresented categories")
        return llm_labeled_docs, 0

    # Combine original sample with augmented docs
    all_labeled = llm_labeled_docs + augmented_docs
    console.success(f"Augmented sample with {len(augmented_docs)} additional examples")

    return all_labeled, len(augmented_docs)


async def label_documents(
    state: State,
    config: RunnableConfig,
) -> dict:
    """Label documents using LLM + classifier approach.

    Strategy:
    1. LLM labels sampled documents (state.documents)
    2. If more documents exist in state.all_documents:
       - Train RandomForest classifier on embeddings
       - Use classifier to label remaining documents
    3. Return all labeled documents
    """
    configuration = Configuration.from_runnable_config(config)
    console = configuration.get_console()

    # Debug: Show configuration
    console.debug(f"Configuration: model={configuration.model}, fast_llm={configuration.fast_llm}")
    console.debug(f"Sample size: {configuration.sample_size}, Batch size: {configuration.batch_size}")
    console.debug(f"Documents to label: {len(state.documents)}, Total documents: {len(state.all_documents)}")

    # Get latest taxonomy
    latest_clusters = None
    for clusters in reversed(state.clusters):
        if isinstance(clusters, list) and clusters:
            latest_clusters = clusters
            break

    if not latest_clusters and state.clusters:
        latest_clusters = [state.clusters[-1]] if isinstance(state.clusters[-1], dict) else state.clusters[-1]

    if not latest_clusters:
        raise ValueError("No valid clusters found in state")

    # Debug: Show taxonomy categories
    console.debug(f"Taxonomy has {len(latest_clusters)} categories:")
    for cat in latest_clusters:
        console.debug(f"  [{cat['id']}] {cat['name']}")

    # Step 1: Label sampled documents with LLM
    labeling_chain = _setup_classification_chain(configuration)

    # Process documents with progress tracking
    labeled_results = []
    with console.progress(len(state.documents), "Labeling documents with LLM") as advance:
        for doc in state.documents:
            result = await labeling_chain.ainvoke(
                {
                    "content": doc["content"] if isinstance(doc, dict) else doc.content,
                    "taxonomy": _format_taxonomy(latest_clusters),
                }
            )
            labeled_results.append(result)
            advance()

    # Create labeled Doc objects for sampled documents
    # Map category IDs to category names
    llm_labeled_docs = []
    warnings_list = []
    other_count = 0

    for doc, category_result in zip(state.documents, labeled_results):
        category_id = category_result.get("category_id")

        if category_id is None:
            warning_msg = f"No category ID returned for doc {doc['id'] if isinstance(doc, dict) else doc.id}, using 'Other'"
            console.warning(warning_msg)
            warnings_list.append(warning_msg)
            category_name = "Other"
            other_count += 1
        else:
            try:
                category_name = _get_category_name_by_id(category_id, latest_clusters)
            except ValueError as e:
                warning_msg = f"{e}, using 'Other'"
                console.warning(warning_msg)
                warnings_list.append(warning_msg)
                category_name = "Other"
                other_count += 1

        llm_labeled_docs.append(Doc(
            id=doc["id"] if isinstance(doc, dict) else doc.id,
            content=doc["content"] if isinstance(doc, dict) else doc.content,
            summary=doc.get("summary", "") if isinstance(doc, dict) else (doc.summary or ""),
            explanation=None,
            category=category_name
        ))

    # Step 2: Check if we need to label more documents
    total_docs = len(state.all_documents)
    sampled_docs = len(state.documents)

    # Calculate sample distribution metrics for diagnosing imbalance
    sample_distribution = Counter(doc.category for doc in llm_labeled_docs if doc.category != "Other")
    all_categories = {cat["name"] for cat in latest_clusters}
    zero_sample_categories = list(all_categories - set(sample_distribution.keys()))

    if sampled_docs >= total_docs:
        # All documents were sampled and labeled by LLM
        console.success(f"All {total_docs} documents labeled by LLM")

        return {
            "documents": llm_labeled_docs,
            "status": [f"All {total_docs} documents labeled by LLM"],
            "llm_labeled_count": len(llm_labeled_docs),
            "classifier_labeled_count": 0,
            "skipped_document_count": other_count,
            "warnings": warnings_list,
            "sample_distribution": dict(sample_distribution),
            "zero_sample_categories": zero_sample_categories,
        }

    # Step 3: Train classifier and label remaining documents
    remaining_count = total_docs - sampled_docs
    console.info(f"Training classifier on {sampled_docs} LLM-labeled documents...")
    console.info(f"  Will classify {remaining_count} remaining documents")

    # Initialize embeddings encoder
    encoder = OpenAIEmbeddings(model=configuration.embedding_model)

    # Step 3.1: Balance sample if min_examples_per_category is set
    augmented_count = 0
    min_per_cat = configuration.min_examples_per_category

    if min_per_cat > 0:
        # Get unlabeled pool for balancing
        labeled_ids = {doc.id for doc in llm_labeled_docs}
        unlabeled_pool_for_balancing = [
            doc for doc in state.all_documents
            if (doc["id"] if isinstance(doc, dict) else doc.id) not in labeled_ids
        ]

        llm_labeled_docs, augmented_count = await _balance_sample(
            llm_labeled_docs,
            unlabeled_pool_for_balancing,
            latest_clusters,
            min_per_cat,
            labeling_chain,
            encoder,
            console,
        )

        # Update sample distribution after balancing
        sample_distribution = Counter(doc.category for doc in llm_labeled_docs if doc.category != "Other")
        zero_sample_categories = list(all_categories - set(sample_distribution.keys()))

    # Generate embeddings for LLM-labeled documents (training set)
    with console.status("Generating embeddings for training set..."):
        training_contents = [doc.content for doc in llm_labeled_docs]
        training_embeddings = await encoder.aembed_documents(training_contents)

    # Train classifier
    with console.status("Training RandomForest classifier..."):
        model, index_to_category, metrics = train_classifier(
            llm_labeled_docs,
            training_embeddings,
            latest_clusters,
            console=console,
        )

    console.success(
        f"Classifier trained - Test F1: {metrics['test_f1']:.3f}, "
        f"Test Accuracy: {metrics['test_accuracy']:.3f}"
    )
    console.debug(f"Classifier metrics detail:")
    console.debug(f"  Train Accuracy: {metrics['train_accuracy']:.3f}, Train F1: {metrics['train_f1']:.3f}")
    console.debug(f"  Test Accuracy: {metrics['test_accuracy']:.3f}, Test F1: {metrics['test_f1']:.3f}")

    # Get unlabeled documents (those not in the sample)
    sampled_ids = {doc.id for doc in llm_labeled_docs}
    unlabeled_docs = [
        doc for doc in state.all_documents
        if (doc["id"] if isinstance(doc, dict) else doc.id) not in sampled_ids
    ]

    # Generate embeddings for unlabeled documents
    with console.status(f"Generating embeddings for {len(unlabeled_docs)} documents..."):
        unlabeled_contents = [
            doc["content"] if isinstance(doc, dict) else doc.content
            for doc in unlabeled_docs
        ]
        unlabeled_embeddings = await encoder.aembed_documents(unlabeled_contents)

    # Predict categories
    with console.status("Classifying with trained model..."):
        predicted_categories = predict_with_classifier(
            model,
            unlabeled_embeddings,
            index_to_category
        )

    # Get confidence scores and handle low-confidence predictions
    threshold = configuration.classifier_confidence_threshold
    low_confidence_action = configuration.low_confidence_action
    llm_relabel_count = 0
    other_relabel_count = 0
    max_llm_relabel = 20  # Safeguard: max docs to re-label with LLM

    if threshold > 0:
        confidences = get_prediction_confidence(model, unlabeled_embeddings)
        low_conf_indices = [i for i, conf in enumerate(confidences) if conf < threshold]

        if low_conf_indices:
            console.info(f"Found {len(low_conf_indices)} low-confidence predictions (below {threshold})")

            if low_confidence_action == "other":
                # Label low-confidence predictions as "Other"
                for idx in low_conf_indices:
                    predicted_categories[idx] = "Other"
                    other_relabel_count += 1
                console.info(f"Labeled {other_relabel_count} low-confidence docs as 'Other'")

            elif low_confidence_action == "llm":
                # Re-label with LLM, but with safeguard
                if len(low_conf_indices) > max_llm_relabel:
                    console.warning(
                        f"Too many low-confidence docs ({len(low_conf_indices)}) for LLM re-labeling. "
                        f"Max is {max_llm_relabel}. Falling back to labeling as 'Other'."
                    )
                    for idx in low_conf_indices:
                        predicted_categories[idx] = "Other"
                        other_relabel_count += 1
                    console.info(f"Labeled {other_relabel_count} low-confidence docs as 'Other'")
                else:
                    console.info(f"Re-labeling {len(low_conf_indices)} low-confidence predictions with LLM...")
                    for idx in low_conf_indices:
                        doc = unlabeled_docs[idx]
                        content = doc["content"] if isinstance(doc, dict) else doc.content
                        result = await labeling_chain.ainvoke({
                            "content": content,
                            "taxonomy": _format_taxonomy(latest_clusters),
                        })
                        category_id = result.get("category_id")
                        if category_id is not None:
                            try:
                                predicted_categories[idx] = _get_category_name_by_id(category_id, latest_clusters)
                                llm_relabel_count += 1
                            except ValueError:
                                predicted_categories[idx] = "Other"
                                other_relabel_count += 1
                        else:
                            predicted_categories[idx] = "Other"
                            other_relabel_count += 1
                    console.success(f"Re-labeled {llm_relabel_count} docs with LLM, {other_relabel_count} as 'Other'")

            elif low_confidence_action == "keep":
                # Keep classifier predictions (do nothing)
                console.info(f"Keeping classifier predictions for {len(low_conf_indices)} low-confidence docs")

    # Create Doc objects for classifier-labeled documents
    classifier_labeled_docs = [
        Doc(
            id=doc["id"] if isinstance(doc, dict) else doc.id,
            content=doc["content"] if isinstance(doc, dict) else doc.content,
            summary=doc.get("summary", "") if isinstance(doc, dict) else (doc.summary or ""),
            explanation=None,
            category=category
        )
        for doc, category in zip(unlabeled_docs, predicted_categories)
    ]

    # Combine all labeled documents
    all_labeled_docs = llm_labeled_docs + classifier_labeled_docs

    console.success(f"Total labeled: {len(all_labeled_docs)} documents")
    console.info(f"  - {len(llm_labeled_docs)} by LLM")
    console.info(f"  - {len(classifier_labeled_docs)} by classifier")

    return {
        "documents": all_labeled_docs,
        "status": [
            f"Labeled {len(llm_labeled_docs)} documents with LLM",
            f"Trained classifier (F1: {metrics['test_f1']:.3f})",
            f"Classified {len(classifier_labeled_docs)} documents with model",
            f"Total: {len(all_labeled_docs)} documents labeled"
        ],
        "classifier_metrics": metrics,
        "llm_labeled_count": len(llm_labeled_docs),
        "classifier_labeled_count": len(classifier_labeled_docs),
        "llm_relabel_count": llm_relabel_count,
        "augmented_count": augmented_count,
        "skipped_document_count": other_count,
        "warnings": warnings_list,
        "sample_distribution": dict(sample_distribution),
        "zero_sample_categories": zero_sample_categories,
        # Store classifier for later export
        "classifier_model": model,
        "classifier_index_to_category": index_to_category,
    }
