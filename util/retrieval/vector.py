def build_vector_retriever(index, similarity_top_k):
    """Build the semantic search side of hybrid retrieval."""
    # Vector search uses embeddings, so it can find chunks with similar meaning
    # even when the user's words do not exactly match the document text.
    return index.as_retriever(
        similarity_top_k=similarity_top_k,
    )
