import os
import shutil

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from util.pdf_utils import load_pdf_documents


INDEX_VERSION = "pdf-pages-v1"


def load_or_create_index(base_dir, pdf_path, persist_dir):
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        if is_current_index_version(persist_dir):
            print("Loading existing index...")

            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir
            )

            return load_index_from_storage(storage_context)

        print("Existing index is outdated. Rebuilding with page metadata...")
        reset_persist_dir(base_dir, persist_dir)

    print("Creating new index...")

    documents = load_pdf_documents(pdf_path)

    index = VectorStoreIndex.from_documents(
        documents
    )

    index.storage_context.persist(
        persist_dir=persist_dir
    )
    write_index_version(persist_dir)

    return index


def is_current_index_version(persist_dir):
    version_path = get_index_version_path(persist_dir)

    if not os.path.exists(version_path):
        return False

    with open(version_path, "r", encoding="utf-8") as f:
        return f.read().strip() == INDEX_VERSION


def write_index_version(persist_dir):
    os.makedirs(persist_dir, exist_ok=True)

    with open(get_index_version_path(persist_dir), "w", encoding="utf-8") as f:
        f.write(INDEX_VERSION)


def reset_persist_dir(base_dir, persist_dir):
    persist_dir = os.path.abspath(persist_dir)
    base_dir = os.path.abspath(base_dir)

    if persist_dir == base_dir or os.path.commonpath([base_dir, persist_dir]) != base_dir:
        raise ValueError(
            "Refusing to rebuild index because PERSIST_DIR is outside the project."
        )

    shutil.rmtree(persist_dir)


def get_index_version_path(persist_dir):
    return os.path.join(persist_dir, ".index_version")
