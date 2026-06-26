import hashlib
import json
import os
import shutil

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import TextNode
from util.pdf_utils import load_pdf_documents


INDEX_VERSION = "pdf-pages-nodes-v1"
NODES_FILE_NAME = "nodes.json"
INDEX_METADATA_FILE_NAME = "index_metadata.json"


def load_or_create_index(base_dir, pdf_path, persist_dir, index_config=None):
    """Load the persisted vector index, or rebuild all index artifacts."""
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        if is_current_index(persist_dir, pdf_path, index_config):
            print("Loading existing index...")

            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir
            )

            return load_index_from_storage(storage_context)

        print("Existing index is outdated. Rebuilding with page metadata...")
        reset_persist_dir(base_dir, persist_dir)

    print("Creating new index...")

    # Documents are the page-level source records from the PDF.
    # Nodes are the smaller searchable chunks produced by LlamaIndex's parser.
    documents = load_pdf_documents(pdf_path)
    nodes = Settings.node_parser.get_nodes_from_documents(documents)

    # Build the vector index from the exact nodes we also persist for BM25.
    # This keeps semantic and keyword retrieval aligned to the same chunks.
    index = VectorStoreIndex(
        nodes
    )

    index.storage_context.persist(
        persist_dir=persist_dir
    )
    write_nodes(persist_dir, nodes)
    write_index_version(persist_dir)
    write_index_metadata(persist_dir, pdf_path, index_config, nodes)

    return index


def load_persisted_nodes(persist_dir):
    """Load the chunked nodes shared by vector and BM25 retrieval."""
    with open(get_nodes_path(persist_dir), "r", encoding="utf-8") as f:
        nodes = json.load(f)

    return [TextNode.from_dict(node) for node in nodes]


def write_nodes(persist_dir, nodes):
    """Persist chunked nodes so BM25 does not need to re-read the PDF."""
    os.makedirs(persist_dir, exist_ok=True)

    with open(get_nodes_path(persist_dir), "w", encoding="utf-8") as f:
        json.dump(
            [node.to_dict() for node in nodes],
            f,
            ensure_ascii=False,
            indent=2,
        )


def is_current_index(persist_dir, pdf_path, index_config=None):
    """Check every persisted artifact needed for a reusable index."""
    return (
        is_current_index_version(persist_dir)
        and os.path.exists(get_nodes_path(persist_dir))
        and is_current_index_metadata(persist_dir, pdf_path, index_config)
    )


def is_current_index_metadata(persist_dir, pdf_path, index_config=None):
    """Check whether source PDF and indexing settings still match storage."""
    metadata_path = get_index_metadata_path(persist_dir)

    if not os.path.exists(metadata_path):
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected_metadata = build_index_metadata(
        pdf_path=pdf_path,
        index_config=index_config,
    )
    comparable_metadata = dict(metadata)
    # Node count is useful to inspect, but the real invalidation inputs are
    # version, PDF hash/path, and index configuration.
    comparable_metadata.pop("node_count", None)

    return comparable_metadata == expected_metadata


def write_index_metadata(persist_dir, pdf_path, index_config, nodes):
    """Write the metadata that decides whether storage is stale later."""
    metadata = build_index_metadata(
        pdf_path=pdf_path,
        index_config=index_config,
    )
    metadata["node_count"] = len(nodes)

    with open(get_index_metadata_path(persist_dir), "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_index_metadata(pdf_path, index_config=None):
    """Build stable metadata for cache invalidation."""
    return {
        "version": INDEX_VERSION,
        "pdf_path": os.path.abspath(pdf_path),
        "pdf_sha256": get_file_sha256(pdf_path),
        "index_config": index_config or {},
    }


def get_file_sha256(file_path):
    """Hash the source PDF so content changes trigger a rebuild."""
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def is_current_index_version(persist_dir):
    """Check the storage schema version."""
    version_path = get_index_version_path(persist_dir)

    if not os.path.exists(version_path):
        return False

    with open(version_path, "r", encoding="utf-8") as f:
        return f.read().strip() == INDEX_VERSION


def write_index_version(persist_dir):
    """Write a small schema marker for the persisted index layout."""
    os.makedirs(persist_dir, exist_ok=True)

    with open(get_index_version_path(persist_dir), "w", encoding="utf-8") as f:
        f.write(INDEX_VERSION)


def reset_persist_dir(base_dir, persist_dir):
    """Delete stale generated storage after verifying it is inside the project."""
    persist_dir = os.path.abspath(persist_dir)
    base_dir = os.path.abspath(base_dir)

    if persist_dir == base_dir or os.path.commonpath([base_dir, persist_dir]) != base_dir:
        raise ValueError(
            "Refusing to rebuild index because PERSIST_DIR is outside the project."
        )

    shutil.rmtree(persist_dir)


def get_index_version_path(persist_dir):
    return os.path.join(persist_dir, ".index_version")


def get_nodes_path(persist_dir):
    return os.path.join(persist_dir, NODES_FILE_NAME)


def get_index_metadata_path(persist_dir):
    return os.path.join(persist_dir, INDEX_METADATA_FILE_NAME)
