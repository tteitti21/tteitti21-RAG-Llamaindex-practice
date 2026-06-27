from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import Memory
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from util.keyword_retrieval_utils import build_keyword_retriever
from util.rerank_utils import rerank_list_section_matches
from util.vector_retrieval_utils import build_vector_retriever


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
        retriever=build_vector_retriever(
            index=index,
            similarity_top_k=retrieval_top_k,
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
