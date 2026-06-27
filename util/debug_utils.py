from colorama import Fore, Style


def print_debug_sources(response, debug_enabled, query_engine=None):
    if not debug_enabled:
        return

    print_retrieval_debug(query_engine)

    print(f"{Fore.CYAN}\nSources:\n")
    for i, source_node in enumerate(response.source_nodes, start=1):
        print(f"\n--- Source {i} ---")
        print(f"{Fore.YELLOW}Score:{Style.RESET_ALL}", source_node.score)
        print(f"{Fore.LIGHTMAGENTA_EX}Metadata:{Style.RESET_ALL}", source_node.metadata)
        print(
            f"{Fore.LIGHTBLUE_EX}Content:{Style.RESET_ALL}",
            source_node.node.get_content()[:700],
        )
        print(f"{Fore.CYAN}" + "_" * 50)


def print_retrieval_debug(query_engine):
    """Print raw retriever rankings and final fused rankings when available."""
    if not query_engine:
        return

    retriever = get_debug_retriever(query_engine)

    if not retriever:
        return

    print(f"{Fore.CYAN}\nRetrieval debug:\n")
    for child_retriever in getattr(retriever, "_retrievers", []):
        print_ranked_nodes(
            title=getattr(child_retriever, "name", "Retriever"),
            nodes=getattr(child_retriever, "last_debug_results", []),
        )

    print_ranked_nodes(
        title="Merged hybrid results",
        nodes=getattr(retriever, "last_fused_debug_results", []),
    )


def print_ranked_nodes(title, nodes):
    """Print a compact ranking list with page, score, and content preview."""
    print(f"{Fore.LIGHTCYAN_EX}{title}:{Style.RESET_ALL}")

    if not nodes:
        print(f"{Fore.YELLOW}No results returned.{Style.RESET_ALL}\n")
        return

    for i, node_with_score in enumerate(nodes, start=1):
        page, score_text, preview, rerank_text = format_debug_node(node_with_score)
        source_info = (
            f"{Fore.LIGHTBLUE_EX}Page{Style.RESET_ALL} {page} | "
            f"{Fore.YELLOW}Score{Style.RESET_ALL} {score_text} | "
            f"{rerank_text}"
            f"{Fore.LIGHTMAGENTA_EX}Preview:{Style.RESET_ALL}"
        )

        print(f"[{i}] {source_info} {preview}")

    print()


def get_debug_retriever(query_engine):
    """Return the retriever from either a query engine or chat engine."""
    if hasattr(query_engine, "retriever"):
        return query_engine.retriever

    return getattr(query_engine, "_retriever", None)


def format_debug_node(node):
    """Format either a debug snapshot or a LlamaIndex NodeWithScore."""
    if isinstance(node, dict):
        score = node["score"]

        return (
            node["page"],
            format_score(score),
            node["preview"],
            format_rerank_boosts(node),
        )

    metadata = node.metadata
    page = metadata.get("page_label") or metadata.get("page") or "unknown"
    preview = " ".join(node.node.get_content().split())[:90]

    return page, format_score(node.score), preview, ""


def format_score(score):
    """Render scores consistently while preserving missing values."""
    if isinstance(score, float):
        return f"{score:.4f}"

    return str(score)


def format_rerank_boosts(node):
    """Render final rerank boosts when a debug snapshot includes them."""
    boost_parts = []

    for label, key in [
        ("Numbered ref", "numbered_reference_boost"),
        ("List section", "list_section_boost"),
    ]:
        boost = node.get(key, 0.0)

        if boost:
            boost_parts.append(
                format_rerank_boost(
                    label,
                    boost,
                    node,
                )
            )

    if not boost_parts:
        return ""

    return f"{Fore.GREEN}Rerank{Style.RESET_ALL} {', '.join(boost_parts)} | "


def format_rerank_boost(label, boost, node):
    """Render one rerank boost with optional match-type details."""
    boost_text = f"{label} +{format_score(boost)}"

    if label != "Numbered ref":
        return boost_text

    match_types = node.get("numbered_reference_match_types", [])

    if not match_types:
        return boost_text

    return f"{boost_text} ({', '.join(match_types)})"
