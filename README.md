# LlamaIndex RAG Practice

This is a small practice project for learning how to build a basic Retrieval-Augmented Generation (RAG) workflow with LlamaIndex.

The app loads a PDF document, creates a vector index from it, persists that index locally, and lets you chat about the document from the terminal.

## What This Project Does

- Loads configuration from a local `.env` file.
- Reads a PDF from the `docs/` folder.
- Uses OpenAI for the LLM and embedding model.
- Builds a LlamaIndex `VectorStoreIndex`.
- Uses hybrid retrieval with vector search and local BM25-style keyword search.
- Saves the vector index, chunked nodes, and BM25 token statistics locally.
- Reuses the saved index on later runs.
- Keeps chat history during a terminal session for follow-up questions.
- Optionally prints source chunks, metadata, and similarity scores for debugging.

## Project Structure

```text
.
+-- docs/
|   +-- ____.pdf
+-- util/
|   +-- debug_utils.py
|   +-- env_utils.py
|   +-- index_utils.py
|   +-- pdf_utils.py
|   +-- retrieval_utils.py
+-- .env.example
+-- .gitignore
+-- llamaindex_rag.py
+-- README.md
+-- requirements.txt
```

### Environment Variables

`OPENAI_API_KEY`

Your OpenAI API key. This is required for both the chat model and embedding model.

`PDF_PATH`

Path to the PDF file that LlamaIndex should read. Relative paths are resolved from the project root.

`PERSIST_DIR`

Directory where generated storage files are saved. The app stores the vector index, chunked nodes, and BM25 keyword statistics there so they can be loaded again instead of rebuilt every time.

`DEBUG`

Set to `true` to print source nodes, similarity scores, metadata, and content snippets after each query. Keep it as `false` for normal answer-only output.

`CHUNK_SIZE`

Token target for each indexed text chunk. Changing this requires rebuilding the persisted index.

`CHUNK_OVERLAP`

Token overlap between neighboring chunks. Changing this requires rebuilding the persisted index.

`RETRIEVAL_TOP_K`

Number of candidate chunks each search method retrieves before fusion. Raising this can help when the right page is ranked slightly below the first few matches.

`LLM_CONTEXT_TOP_K`

Number of final fused chunks to send into answer generation.
