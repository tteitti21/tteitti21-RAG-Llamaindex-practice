import re

from util.retrieval.keyword import (
    tokenize,
)
from util.retrieval.references import (
    get_numbered_reference_boost,
    get_numbered_reference_debug,
    has_numbered_reference,
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
        **get_numbered_reference_debug(query_tokens, node),
        "list_section_boost": get_list_section_boost(
            query_tokens,
            node,
        ),
    }


def get_list_section_boost(query_tokens, node):
    """Boost front-matter list pages for contents, figures, and tables."""
    section_headings = get_list_section_headings(query_tokens)

    if not section_headings:
        return 0.0

    content = " ".join(node.get_content().lower().split())

    for heading in section_headings:
        if content.startswith(heading):
            return 6.0

        # List headings often appear after the main contents list has continued
        # across pages, as with "Kuvat" and "Taulukot" in this PDF.
        if re.search(rf"(^|\s){re.escape(heading)}\s+", content):
            return 4.0

    return 0.0


def get_list_section_headings(query_tokens):
    """Map query intent tokens to front-matter headings in the document."""
    if has_numbered_reference(query_tokens):
        return []

    headings = []

    for intent_token, section_headings in LIST_SECTION_INTENTS.items():
        if intent_token in query_tokens:
            headings.extend(section_headings)

    return headings


LIST_SECTION_INTENTS = {
    "sisällysluettelo": ["sisältö"],
    "sisältö": ["sisältö"],
    "sisälö": ["sisältö"],
    "kuv": ["kuvat"],
    "kuva": ["kuvat"],
    "kuvaluettelo": ["kuvat"],
    "tauluko": ["taulukot"],
    "taulukkoluettelo": ["taulukot"],
}
"""Map normalized query tokens to front-matter section headings.

The keys are tokens produced by tokenize(). The values are raw lowercase
headings that may appear at the start of front-matter pages, such as "sisältö",
"kuvat", or "taulukot".
"""
