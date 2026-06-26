import math
import re
from collections import Counter

from llama_index.core import QueryBundle, Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore
from util.pdf_utils import load_pdf_documents


def build_hybrid_query_engine(index, pdf_path, similarity_top_k, qa_prompt):
    """Build a query engine that combines semantic and lexical retrieval."""
    vector_retriever = index.as_retriever(
        similarity_top_k=similarity_top_k
    )
    keyword_retriever = build_keyword_retriever(
        pdf_path=pdf_path,
        similarity_top_k=similarity_top_k,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=similarity_top_k,
        # Keep this at one query to avoid extra LLM calls for query rewriting.
        num_queries=1,
        use_async=False,
        # Prefer semantic search slightly, while still letting exact keywords win.
        retriever_weights=[0.6, 0.4],
    )

    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        text_qa_template=qa_prompt,
    )


def build_keyword_retriever(pdf_path, similarity_top_k):
    """Create a local BM25 retriever from the same PDF pages as the vector index."""
    nodes = Settings.node_parser.get_nodes_from_documents(
        load_pdf_documents(pdf_path),
    )

    return BM25Retriever(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
    )


class BM25Retriever(BaseRetriever):
    """Small local BM25 retriever used as the keyword side of hybrid search."""

    def __init__(self, nodes, similarity_top_k, k1=1.5, b=0.75):
        """Pre-tokenize nodes and collect corpus statistics for BM25 scoring."""
        super().__init__()
        self.nodes = nodes
        self.similarity_top_k = similarity_top_k
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(node.get_content()) for node in nodes]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_frequencies = self._build_doc_frequencies()

    def _retrieve(self, query_bundle: QueryBundle):
        """Return the highest-scoring nodes for the query using BM25."""
        query_tokens = tokenize(query_bundle.query_str)
        scored_nodes = []

        for node, doc_tokens, doc_length in zip(
            self.nodes,
            self.doc_tokens,
            self.doc_lengths,
        ):
            score = self._score(query_tokens, doc_tokens, doc_length)

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

    def _build_doc_frequencies(self):
        """Count how many documents contain each token."""
        doc_frequencies = Counter()

        for tokens in self.doc_tokens:
            doc_frequencies.update(set(tokens))

        return doc_frequencies

    def _score(self, query_tokens, doc_tokens, doc_length):
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

        return score


def tokenize(text):
    """Split text into normalized keyword tokens for lexical retrieval."""
    tokens = []

    for token in re.findall(r"\w+", text.lower()):
        if token in STOPWORDS:
            continue

        normalized_token = normalize_token(token)

        if normalized_token and normalized_token not in STOPWORDS:
            tokens.append(normalized_token)

    return tokens


def normalize_token(token):
    """Apply light Finnish suffix stripping for better word-form matching."""
    for suffix in FINNISH_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]

    return token


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
    "sanottiin",
    "sanottii",
}

FINNISH_SUFFIXES = [
    "isissa",
    "isissä",
    "ista",
    "istä",
    "ssa",
    "ssä",
    "sta",
    "stä",
    "lla",
    "llä",
    "lle",
    "ksi",
    "kin",
    "kaan",
    "kään",
    "een",
    "den",
    "ten",
    "tta",
    "ttä",
    "ta",
    "tä",
    "na",
    "nä",
    "n",
    "a",
    "ä",
    "t",
]
