import json
import os
import re

import Stemmer
import bm25s
from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever
from util.index_utils import load_persisted_nodes
from util.retrieval.references import has_numbered_reference


BM25_INDEX_DIR_NAME = "llama_bm25"
BM25_METADATA_FILE_NAME = "metadata.json"
BM25_INDEX_VERSION = "llama-bm25-finnish-v1"
BM25_TOKEN_PATTERN = r"(?u)\b\w+\b"


def build_keyword_retriever(persist_dir, similarity_top_k):
    """Create a LlamaIndex BM25 retriever from persisted vector index nodes."""
    # Loading persisted nodes keeps BM25 and vector retrieval anchored to the
    # same chunk boundaries and metadata.
    nodes = load_persisted_nodes(persist_dir)

    return QueryExpansionRetriever(
        retriever=load_or_create_bm25_retriever(
            persist_dir=persist_dir,
            nodes=nodes,
            similarity_top_k=similarity_top_k,
        )
    )


def load_or_create_bm25_retriever(persist_dir, nodes, similarity_top_k):
    """Load a persisted LlamaIndex BM25 retriever, or build and persist one."""
    bm25_index_dir = get_bm25_index_dir(persist_dir)
    bm25_metadata_path = get_bm25_metadata_path(persist_dir)

    if os.path.exists(bm25_metadata_path):
        with open(bm25_metadata_path, "r", encoding="utf-8") as f:
            bm25_metadata = json.load(f)

        if is_current_bm25_index(bm25_metadata, nodes):
            return load_persisted_bm25_retriever(
                bm25_index_dir=bm25_index_dir,
                similarity_top_k=similarity_top_k,
            )

    retriever = LlamaBM25Retriever.from_defaults(
        nodes=nodes,
        stemmer=get_finnish_stemmer(),
        language=list(BM25_STOPWORDS),
        similarity_top_k=similarity_top_k,
        token_pattern=BM25_TOKEN_PATTERN,
    )

    os.makedirs(bm25_index_dir, exist_ok=True)
    # Persist through bm25s directly so we can restore with the same Finnish
    # stemmer and single-character token pattern that LlamaIndex does not store.
    retriever.bm25.save(
        bm25_index_dir,
        corpus=retriever.corpus,
        show_progress=False,
    )

    with open(bm25_metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            build_bm25_metadata(nodes),
            f,
            ensure_ascii=False,
            indent=2,
        )

    return retriever


def load_persisted_bm25_retriever(bm25_index_dir, similarity_top_k):
    """Restore a LlamaIndex BM25 retriever with project tokenizer settings."""
    bm25 = bm25s.BM25.load(
        bm25_index_dir,
        load_corpus=True,
    )

    return LlamaBM25Retriever(
        existing_bm25=bm25,
        stemmer=get_finnish_stemmer(),
        language=list(BM25_STOPWORDS),
        similarity_top_k=similarity_top_k,
        token_pattern=BM25_TOKEN_PATTERN,
    )


def is_current_bm25_index(bm25_metadata, nodes):
    """Check that cached BM25 data matches the current node cache."""
    # Node hashes catch content or metadata changes even if the node count stays
    # the same.
    return (
        bm25_metadata.get("version") == BM25_INDEX_VERSION
        and bm25_metadata.get("node_count") == len(nodes)
        and bm25_metadata.get("node_hashes") == get_node_hashes(nodes)
        and bm25_metadata.get("token_pattern") == BM25_TOKEN_PATTERN
    )


def build_bm25_metadata(nodes):
    """Build metadata used to validate the persisted LlamaIndex BM25 index."""
    return {
        "version": BM25_INDEX_VERSION,
        "node_count": len(nodes),
        "node_hashes": get_node_hashes(nodes),
        "token_pattern": BM25_TOKEN_PATTERN,
    }


def get_node_hashes(nodes):
    return [node.hash for node in nodes]


def get_bm25_index_dir(persist_dir):
    return os.path.join(persist_dir, BM25_INDEX_DIR_NAME)


def get_bm25_metadata_path(persist_dir):
    return os.path.join(get_bm25_index_dir(persist_dir), BM25_METADATA_FILE_NAME)


def get_finnish_stemmer():
    return Stemmer.Stemmer("finnish")


class QueryExpansionRetriever(BaseRetriever):
    """Expand Finnish query terms before delegating to LlamaIndex BM25."""

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever

    def _retrieve(self, query_bundle: QueryBundle):
        # LlamaIndex BM25 handles scoring and indexing; this wrapper only keeps
        # the project-specific Finnish synonyms available for query text.
        expanded_query = build_bm25_query(query_bundle.query_str)
        bm25_query = expanded_query or query_bundle.query_str

        return [
            node
            for node in self.retriever.retrieve(QueryBundle(bm25_query))
            if (node.score or 0.0) > 0
        ]


def build_bm25_query(text):
    """Expand a query with raw words that PyStemmer can safely stem once."""
    raw_tokens = [
        token
        for token in re.findall(r"\w+", text.lower())
        if token not in BM25_STOPWORDS
    ]
    normalized_tokens = tokenize(text)
    expanded_tokens = []

    for token in raw_tokens:
        if token not in expanded_tokens:
            expanded_tokens.append(token)

    for token in normalized_tokens:
        for synonym in BM25_QUERY_SYNONYMS.get(token, []):
            if synonym not in expanded_tokens:
                expanded_tokens.append(synonym)

    return " ".join(expanded_tokens)


def tokenize(text):
    """Split text into normalized keyword tokens for lexical retrieval."""
    tokens = []

    for token in re.findall(r"\w+", text.lower()):
        if token in STOPWORDS:
            continue

        normalized_token = normalize_token(token)

        if normalized_token and normalized_token not in STOPWORDS:
            tokens.extend(expand_token(normalized_token))

    return tokens


def normalize_token(token):
    """Stem Finnish tokens with the same PyStemmer family used by BM25."""
    return get_finnish_stemmer().stemWord(token)


def expand_token(token):
    """Add domain synonyms that stemming cannot discover on its own."""
    expanded_tokens = [token]

    for synonym in RETRIEVAL_SYNONYMS.get(token, []):
        if synonym not in expanded_tokens:
            expanded_tokens.append(synonym)

    return expanded_tokens


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


STOPWORDS = {
    "ja",
    "tai",
    "on",
    "oli",
    "ovat",
    "mitä",
    "mita",
    "joka",
    "jotka",
    "halua",
    "haluais",
    "että",
    "etta",
    "kaik",
    "list",
    "listauks",
    "sekä",
    "seka",
    "tiedosto",
    "kun",
    "kuin",
    "myös",
    "myos",
    "sais",
    "saisinko",
    "sanottiin",
    "sanottii",
}

BM25_STOPWORDS = STOPWORDS | {
    "että",
    "haluan",
    "haluaisin",
    "kaikista",
    "kaikki",
    "listaa",
    "listauksen",
    "mitä",
    "myös",
    "näiden",
    "näistä",
    "saisinko",
    "sekä",
    "tiedosto",
    "tiedoston",
    "tiedostosta",
}

RETRIEVAL_SYNONYMS = {
    "sisällysluettelo": ["sisältö", "sisälö"],
    "sisältö": ["sisällysluettelo", "sisälö"],
    "sisälö": ["sisältö", "sisällysluettelo"],
    "kuvaluettelo": ["kuva", "kuv"],
    "kuv": ["kuva", "kuvaluettelo"],
    "kuva": ["kuv", "kuvaluettelo"],
    "taulukkoluettelo": ["tauluko"],
    "tauluko": ["taulukkoluettelo"],
}

BM25_QUERY_SYNONYMS = {
    "sisällysluettelo": ["sisältö", "sisällöstä"],
    "sisältö": ["sisällysluettelo", "sisällöstä"],
    "sisälö": ["sisältö", "sisällysluettelo"],
    "kuvaluettelo": ["kuva", "kuvat", "kuvista"],
    "kuv": ["kuva", "kuvat", "kuvista", "kuvaluettelo"],
    "kuva": ["kuvat", "kuvista", "kuvaluettelo"],
    "taulukkoluettelo": ["taulukko", "taulukot"],
    "tauluko": ["taulukko", "taulukot", "taulukkoluettelo"],
}

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
