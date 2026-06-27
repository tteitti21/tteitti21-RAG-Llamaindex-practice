from util.retrieval.keyword import (
    get_list_section_boost,
    get_numbered_reference_boost,
    tokenize,
)


def rerank_list_section_matches(query, nodes):
    """Prefer exact numbered references and front-matter list pages."""
    query_tokens = tokenize(query)

    return sorted(
        nodes,
        key=lambda node: (
            get_numbered_reference_boost(query_tokens, node.node),
            get_list_section_boost(query_tokens, node.node),
            node.score or 0.0,
        ),
        reverse=True,
    )


def get_rerank_debug(query, node):
    """Return rerank boost details for debug output."""
    query_tokens = tokenize(query)

    return {
        "numbered_reference_boost": get_numbered_reference_boost(
            query_tokens,
            node,
        ),
        "list_section_boost": get_list_section_boost(
            query_tokens,
            node,
        ),
    }
