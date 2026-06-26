
import os

from llama_index.core import (
    PromptTemplate,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from colorama import Fore, init
from util.debug_utils import print_debug_sources
from util.env_utils import (
    get_env_bool,
    get_env_int,
    get_env_path,
    get_env_value,
    load_env_file,
)
from util.index_utils import load_or_create_index
from util.retrieval_utils import build_hybrid_query_engine

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

CHUNK_SIZE = get_env_int(ENV_VALUES, "CHUNK_SIZE", 1000)
CHUNK_OVERLAP = get_env_int(ENV_VALUES, "CHUNK_OVERLAP", 200)
SIMILARITY_TOP_K = get_env_int(ENV_VALUES, "SIMILARITY_TOP_K", 10)

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

    index = load_or_create_index(
        base_dir=BASE_DIR,
        pdf_path=PDF_PATH,
        persist_dir=PERSIST_DIR,
    )

    query_engine = build_hybrid_query_engine(
        index=index,
        pdf_path=PDF_PATH,
        similarity_top_k=SIMILARITY_TOP_K,
        qa_prompt=QA_PROMPT,
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
