from colorama import Fore, Style


def print_debug_sources(response, debug_enabled):
    if not debug_enabled:
        return

    print(f"{Fore.CYAN}\nSources:\n")
    for i, source_node in enumerate(response.source_nodes, start=1):
        print(f"\n--- Source {i} ---")
        print(f"{Fore.YELLOW}Score:{Style.RESET_ALL}", source_node.score)
        print(f"{Fore.LIGHTMAGENTA_EX}Metadata:{Style.RESET_ALL}", source_node.metadata)
        print(
            f"{Fore.LIGHTBLUE_EX}Content{Style.RESET_ALL}",
            source_node.node.get_content()[:700],
        )
        print(f"{Fore.CYAN}" + "_" * 50)
