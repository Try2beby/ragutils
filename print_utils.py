import pandas as pd
from llama_index.core.schema import ImageNode, MetadataMode, NodeWithScore
from llama_index.core.utils import truncate_text


def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}

    # if include_cohere_rerank:
    #     crr_relevancy = full_df["cohere_rerank_relevancy"].mean()
    #     columns.update({"cohere_rerank_relevancy": [crr_relevancy]})

    metric_df = pd.DataFrame(columns)

    return metric_df


def display_source_node_cmd(
    source_node: NodeWithScore,
    source_length: int = 100,
    show_source_metadata: bool = False,
    metadata_mode: MetadataMode = MetadataMode.NONE,
) -> None:
    """Display source node for command line."""
    source_text_fmt = truncate_text(
        source_node.node.get_content(metadata_mode=metadata_mode).strip(), source_length
    )
    text_md = (
        f"Node ID: {source_node.node.node_id}\n"
        f"Similarity: {source_node.score}\n"
        f"Text: {source_text_fmt}\n"
    )
    if show_source_metadata:
        text_md += f"Metadata: {source_node.node.metadata}\n"
    if isinstance(source_node.node, ImageNode):
        text_md += "Image:\n"

    print(text_md)
    if isinstance(source_node.node, ImageNode) and source_node.node.image is not None:
        print("Image data is not displayable in command line.")
