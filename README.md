# LlamaIndex RAG Practice

This is a small practice project for learning how to build a basic Retrieval-Augmented Generation (RAG) workflow with LlamaIndex.

The app loads a PDF document, creates a vector index from it, persists that index locally, and lets you ask questions about the document from the terminal.

## What This Project Does

- Loads configuration from a local `.env` file.
- Reads a PDF from the `docs/` folder.
- Uses OpenAI for the LLM and embedding model.
- Builds a LlamaIndex `VectorStoreIndex`.
- Saves the index to a local persistence directory.
- Reuses the saved index on later runs.
- Optionally prints source chunks, metadata, and similarity scores for debugging.

## Project Structure

```text
.
+-- docs/
|   +-- ____.pdf
+-- util/
|   +-- debug_utils.py
|   +-- env_utils.py
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

Directory where the generated LlamaIndex storage files are saved. Keeping this persisted means the index can be loaded again instead of rebuilt every time.

`DEBUG`

Set to `true` to print source nodes, similarity scores, metadata, and content snippets after each query. Keep it as `false` for normal answer-only output.
