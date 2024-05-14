from llama_index.core.vector_stores.types import VectorStoreQueryResult


def relative_rank_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    # NOTE: only for hybrid search (0 for sparse search, 1 for dense search)
    alpha: float = 0.5,
    top_k: int = 2,
):
    """
    Apply reciprocal rank fusion.

    The original paper uses k=60 for best results:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    """
    # check if dense or sparse results is empty
    if (dense_result.nodes is None or len(dense_result.nodes) == 0) and (
        sparse_result.nodes is None or len(sparse_result.nodes) == 0
    ):
        return VectorStoreQueryResult(nodes=None, similarities=None, ids=None)
    elif sparse_result.nodes is None or len(sparse_result.nodes) == 0:
        return dense_result
    elif dense_result.nodes is None or len(dense_result.nodes) == 0:
        return sparse_result

    assert dense_result.nodes is not None
    assert dense_result.similarities is not None
    assert sparse_result.nodes is not None
    assert sparse_result.similarities is not None

    # deconstruct results
    sparse_result_tuples = list(zip(sparse_result.similarities, sparse_result.nodes))
    sparse_result_tuples.sort(key=lambda x: x[0], reverse=True)

    dense_result_tuples = list(zip(dense_result.similarities, dense_result.nodes))
    dense_result_tuples.sort(key=lambda x: x[0], reverse=True)

    results = {
        "dense": [x[1] for x in dense_result_tuples],
        "sparse": [x[1] for x in sparse_result_tuples],
    }

    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes in results.values():
        for rank, node in enumerate(nodes):
            text = node.get_content()
            text_to_node[text] = node
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    fused_similarities = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    fused_similarities = fused_similarities[:top_k]

    # create final response object
    return VectorStoreQueryResult(
        nodes=[text_to_node[x[0]] for x in fused_similarities],
        similarities=[x[1] for x in fused_similarities],
        ids=[text_to_node[x[0]].node_id for x in fused_similarities],
    )
