
import os

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from util.debug_utils import print_debug_sources
from util.env_utils import get_env_bool, get_env_path, get_env_value, load_env_file
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from colorama import Fore, init
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
ENV_VALUES = load_env_file(ENV_PATH)
PDF_PATH = get_env_path(ENV_VALUES, "PDF_PATH", BASE_DIR)
PERSIST_DIR = get_env_path(ENV_VALUES, "PERSIST_DIR", BASE_DIR)
DEBUG = get_env_bool(ENV_VALUES, "DEBUG", False)
OPENAI_API_KEY = get_env_value(ENV_VALUES, "OPENAI_API_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
init(autoreset=True)  # Automatically resets style after every print

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

QA_PROMPT = PromptTemplate(
    """
You are a document-based assistant.

Answer the question using only the provided context.

You may make reasonable inferences if they are directly supported by the context.
Do not introduce information that is not supported by the context.

When possible, cite the source page or source metadata for the answer.
Use short citations like: [page 12]

If the answer cannot be determined from the context, say:
"I cannot find that information in the provided documents."

Context:
{context_str}

Question:
{query_str}

Answer:
"""
)

def load_or_create_index():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("Loading existing index...")

        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR
        )

        return load_index_from_storage(storage_context)

    print("Creating new index...")

    documents = SimpleDirectoryReader(
        input_files=[PDF_PATH]
    ).load_data()

    index = VectorStoreIndex.from_documents(
        documents
    )

    index.storage_context.persist(
        persist_dir=PERSIST_DIR
    )

    return index



def main():
    if not get_env_value(ENV_VALUES, "OPENAI_API_KEY"):
        raise ValueError(
            "Missing OPENAI_API_KEY. Add it to .env, for example: "
            "OPENAI_API_KEY=sk-your-key-here"
        )

    Settings.llm = OpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
    )

    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    index = load_or_create_index()

    query_engine = index.as_query_engine(
        similarity_top_k=5
    )
    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": QA_PROMPT
        }
    )

    while True:
        print(f"{Fore.CYAN}" + "_" * 50)
        question = input("\nQuestion: ")

        if question.lower() in ["exit", "quit"]:
            break

        response = query_engine.query(question)

        print_debug_sources(response, DEBUG)

        print(f"{Fore.GREEN}Answer:")
        print(response)

        print(f"{Fore.CYAN}\nSources used:")
        for i, source_node in enumerate(response.source_nodes, start=1):
            metadata = source_node.metadata

            page = (
                metadata.get("page_label")
                or metadata.get("page")
                or "unknown"
            )

            file_name = metadata.get("file_name", "unknown file")

            print(f"[{i}] {file_name}, page {page}, score {source_node.score:.4f}")


if __name__ == "__main__":
    main()
