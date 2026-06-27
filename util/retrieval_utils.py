import json
import math
import os
import re
from collections import Counter

from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import Memory
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore
from nltk.stem.snowball import SnowballStemmer
from util.index_utils import load_persisted_nodes


BM25_INDEX_FILE_NAME = "bm25_index.json"
BM25_INDEX_VERSION = "bm25-finnish-list-intents-v1"
FINNISH_STEMMER = SnowballStemmer("finnish")


def build_hybrid_query_engine(
    index,
    persist_dir,
    retrieval_top_k,
    llm_context_top_k,
    qa_prompt,
):
    """Build a query engine that combines semantic and lexical retrieval."""
    # Kept for experimentation: this project currently uses the chat engine,
    # but query engines are useful when every question should be standalone.
    hybrid_retriever = build_hybrid_retriever(
        index=index,
        persist_dir=persist_dir,
        retrieval_top_k=retrieval_top_k,
        llm_context_top_k=llm_context_top_k,
    )

    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        text_qa_template=qa_prompt,
    )


def build_hybrid_chat_engine(
    index,
    persist_dir,
    retrieval_top_k,
    llm_context_top_k,
    chat_memory_token_limit,
    context_prompt,
    condense_prompt=None,
):
    """Build a chat engine with memory and the same hybrid retriever."""
    # CondensePlusContextChatEngine rewrites follow-up questions using chat
    # history, then asks the retriever for context using that standalone query.
    hybrid_retriever = build_hybrid_retriever(
        index=index,
        persist_dir=persist_dir,
        retrieval_top_k=retrieval_top_k,
        llm_context_top_k=llm_context_top_k,
    )

    return CondensePlusContextChatEngine.from_defaults(
        retriever=hybrid_retriever,
        memory=Memory.from_defaults(token_limit=chat_memory_token_limit),
        context_prompt=context_prompt,
        condense_prompt=condense_prompt,
        verbose=False,
    )


def build_hybrid_retriever(index, persist_dir, retrieval_top_k, llm_context_top_k):
    """Build the retriever shared by query and chat engines."""
    # Vector search is semantic: it can match similar meaning even when the
    # exact words differ.
    vector_retriever = TrackingRetriever(
        name="Semantic vector search",
        retriever=index.as_retriever(
            similarity_top_k=retrieval_top_k
        ),
    )
    # BM25 is lexical: it rewards exact token matches, which helps with table
    # titles, names, code terms, and page-specific wording.
    keyword_retriever = TrackingRetriever(
        name="BM25 keyword search",
        retriever=build_keyword_retriever(
            persist_dir=persist_dir,
            similarity_top_k=retrieval_top_k,
        ),
    )

    return DebugQueryFusionRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        # Let fusion see the broader candidate pool. DebugQueryFusionRetriever
        # trims to llm_context_top_k after any final intent-aware reranking.
        similarity_top_k=retrieval_top_k,
        final_top_k=llm_context_top_k,
        # Keep this at one query to avoid extra LLM calls for query rewriting.
        num_queries=1,
        use_async=False,
        # Prefer semantic search slightly, while still letting exact keywords win.
        retriever_weights=[0.6, 0.4],
    )


class TrackingRetriever(BaseRetriever):
    """Retriever wrapper that stores the most recent result list for debugging."""

    def __init__(self, name, retriever):
        super().__init__()
        self.name = name
        self.retriever = retriever
        self.last_results = []
        self.last_debug_results = []

    def _retrieve(self, query_bundle: QueryBundle):
        """Delegate retrieval and remember the raw ranked results."""
        self.last_results = self.retriever.retrieve(query_bundle)
        self.last_debug_results = snapshot_results(self.last_results)

        return self.last_results


class DebugQueryFusionRetriever(QueryFusionRetriever):
    """Query fusion retriever that stores the final fused ranking for debugging."""

    def __init__(self, *args, final_top_k=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_top_k = final_top_k
        self.last_fused_results = []
        self.last_fused_debug_results = []

    def _retrieve(self, query_bundle: QueryBundle):
        """Delegate hybrid retrieval and remember the merged ranked results."""
        fused_results = super()._retrieve(query_bundle)
        reranked_results = rerank_list_section_matches(
            query_bundle.query_str,
            fused_results,
        )
        final_top_k = self.final_top_k or len(reranked_results)
        self.last_fused_results = reranked_results[:final_top_k]
        self.last_fused_debug_results = snapshot_results(self.last_fused_results)

        return self.last_fused_results


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


def snapshot_results(nodes):
    """Store result details before fusion mutates node scores in place."""
    snapshots = []

    for node_with_score in nodes:
        metadata = node_with_score.metadata
        snapshots.append(
            {
                "page": metadata.get("page_label") or metadata.get("page") or "unknown",
                "score": node_with_score.score,
                "preview": " ".join(node_with_score.node.get_content().split())[:180],
            }
        )

    return snapshots


def build_keyword_retriever(persist_dir, similarity_top_k):
    """Create a local BM25 retriever from the persisted vector index nodes."""
    # Loading persisted nodes keeps BM25 and vector retrieval anchored to the
    # same chunk boundaries and metadata.
    nodes = load_persisted_nodes(persist_dir)
    bm25_index = load_or_create_bm25_index(persist_dir, nodes)

    return BM25Retriever(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        bm25_index=bm25_index,
    )


def load_or_create_bm25_index(persist_dir, nodes):
    """Load cached BM25 token statistics, or build and persist them."""
    bm25_index_path = get_bm25_index_path(persist_dir)

    if os.path.exists(bm25_index_path):
        with open(bm25_index_path, "r", encoding="utf-8") as f:
            bm25_index = json.load(f)

        if is_current_bm25_index(bm25_index, nodes):
            return bm25_index

    # BM25 statistics are deterministic from nodes + tokenizer, so they can be
    # cached safely and rebuilt only when the nodes or BM25 version change.
    bm25_index = build_bm25_index(nodes)

    with open(bm25_index_path, "w", encoding="utf-8") as f:
        json.dump(
            bm25_index,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return bm25_index


def is_current_bm25_index(bm25_index, nodes):
    """Check that cached BM25 statistics match the current node cache."""
    # Node hashes catch content or metadata changes even if the node count stays
    # the same.
    return (
        bm25_index.get("version") == BM25_INDEX_VERSION
        and bm25_index.get("node_count") == len(nodes)
        and bm25_index.get("node_hashes") == get_node_hashes(nodes)
    )


def build_bm25_index(nodes):
    """Build serializable BM25 token statistics from persisted nodes."""
    # doc_tokens is intentionally stored, not recomputed at startup, because
    # tokenization becomes expensive as the corpus grows.
    doc_tokens = [tokenize(node.get_content()) for node in nodes]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_frequencies = Counter()

    for tokens in doc_tokens:
        doc_frequencies.update(set(tokens))

    return {
        "version": BM25_INDEX_VERSION,
        "node_count": len(nodes),
        "node_hashes": get_node_hashes(nodes),
        "doc_tokens": doc_tokens,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "doc_frequencies": dict(doc_frequencies),
    }


def get_node_hashes(nodes):
    return [node.hash for node in nodes]


def get_bm25_index_path(persist_dir):
    return os.path.join(persist_dir, BM25_INDEX_FILE_NAME)


class BM25Retriever(BaseRetriever):
    """Small local BM25 retriever used as the keyword side of hybrid search."""

    def __init__(self, nodes, similarity_top_k, bm25_index, k1=1.5, b=0.75):
        """Pre-tokenize nodes and collect corpus statistics for BM25 scoring."""
        super().__init__()
        self.nodes = nodes
        self.similarity_top_k = similarity_top_k
        self.k1 = k1
        self.b = b
        self.doc_tokens = bm25_index["doc_tokens"]
        self.doc_lengths = bm25_index["doc_lengths"]
        self.avg_doc_length = bm25_index["avg_doc_length"]
        self.doc_frequencies = Counter(bm25_index["doc_frequencies"])

    def _retrieve(self, query_bundle: QueryBundle):
        """Return the highest-scoring nodes for the query using BM25."""
        # Query text is tokenized with the same rules as the stored documents.
        query_tokens = tokenize(query_bundle.query_str)
        scored_nodes = []

        for node, doc_tokens, doc_length in zip(
            self.nodes,
            self.doc_tokens,
            self.doc_lengths,
        ):
            score = self._score(
                query_tokens=query_tokens,
                doc_tokens=doc_tokens,
                doc_length=doc_length,
                node=node,
            )

            if score > 0:
                scored_nodes.append(
                    NodeWithScore(
                        node=node,
                        score=score,
                    )
                )

        return sorted(
            scored_nodes,
            key=lambda node_with_score: node_with_score.score or 0.0,
            reverse=True,
        )[: self.similarity_top_k]

    def _score(self, query_tokens, doc_tokens, doc_length, node):
        """Calculate the BM25 relevance score for one node."""
        term_frequencies = Counter(doc_tokens)
        score = 0.0
        total_docs = len(self.nodes)

        for token in query_tokens:
            term_frequency = term_frequencies[token]

            if term_frequency == 0:
                continue

            doc_frequency = self.doc_frequencies[token]
            # BM25 rewards rare query terms more than common terms.
            idf = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            # The denominator dampens repeated terms and normalizes long chunks.
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator

        score += get_list_section_boost(query_tokens, node)

        return score


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
    """Stem Finnish tokens with Snowball instead of manual suffix stripping."""
    return FINNISH_STEMMER.stem(token)


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
    "että",
    "etta",
    "sekä",
    "seka",
    "kun",
    "kuin",
    "myös",
    "myos",
    "näyt",
    "näytä",
    "sais",
    "saisinko",
    "sanottiin",
    "sanottii",
}

RETRIEVAL_SYNONYMS = {
    "sisällysluettelo": ["sisältö"],
    "sisältö": ["sisällysluettelo"],
    "sisälö": ["sisältö", "sisällysluettelo"],
    "kuvaluettelo": ["kuva"],
    "kuva": ["kuvaluettelo"],
    "taulukkoluettelo": ["tauluko"],
    "tauluko": ["taulukkoluettelo"],
}

LIST_SECTION_INTENTS = {
    "sisällysluettelo": ["sisältö"],
    "sisältö": ["sisältö"],
    "kuva": ["kuvat"],
    "kuvaluettelo": ["kuvat"],
    "tauluko": ["taulukot"],
    "taulukkoluettelo": ["taulukot"],
}
