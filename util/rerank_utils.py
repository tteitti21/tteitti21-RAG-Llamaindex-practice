from util.keyword_retrieval_utils import get_list_section_boost, tokenize


def rerank_list_section_matches(query, nodes):
    """Prefer exact front-matter list pages for list-style queries."""
    query_tokens = tokenize(query)

    return sorted(
        nodes,
        key=lambda node: (
            get_list_section_boost(query_tokens, node.node),
            node.score or 0.0,
        ),
        reverse=True,
    )
