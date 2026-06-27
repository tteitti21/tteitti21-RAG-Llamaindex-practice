import re


IMAGE_REFERENCE_TOKENS = {
    "kuv",
    "kuva",
    "kuvaluettelo",
}

TABLE_REFERENCE_TOKENS = {
    "tauluk",
    "tauluko",
    "taulukko",
    "taulukkoluettelo",
}

REFERENCE_LABEL_PATTERNS = {
    "kuva": r"kuv\w*",
    "taulukko": r"tauluk\w*",
}

LIST_SECTION_HEADINGS = {
    "kuva": "kuvat",
    "taulukko": "taulukot",
}

REFERENCE_QUERY_BOOSTS = {
    "discussion": {
        "discussion_reference": 12.0,
        "caption": 6.0,
        "list_entry": 1.0,
    },
    "caption": {
        "caption": 12.0,
        "discussion_reference": 6.0,
        "list_entry": 1.0,
    },
    "general": {
        "discussion_reference": 10.0,
        "caption": 10.0,
        "list_entry": 1.0,
    },
}

DISCUSSION_INTENT_TOKENS = {
    "kero",
    "kerro",
    "liitty",
    "selit",
    "tarkemp",
    "tieto",
}

CAPTION_INTENT_TOKENS = {
    "mika",
    "mikä",
    "miss",
    "nayt",
    "näyt",
    "sijait",
}


def has_numbered_reference(query_tokens):
    """Return True when the query asks about a numbered figure or table."""
    return bool(get_numbered_references(query_tokens))


def get_numbered_references(query_tokens):
    """Find numbered figure/table references in normalized query tokens."""
    numbers = [token for token in query_tokens if token.isdigit()]
    references = []

    if not numbers:
        return references

    for number in numbers:
        if any(token in IMAGE_REFERENCE_TOKENS for token in query_tokens):
            references.append(("kuva", number))

        if any(token in TABLE_REFERENCE_TOKENS for token in query_tokens):
            references.append(("taulukko", number))

    return references


def get_numbered_reference_boost(query_tokens, node):
    """Score specific figure/table references by their match type."""
    intent = get_numbered_reference_intent(query_tokens)
    match_types = get_numbered_reference_match_types(query_tokens, node)
    boosts = REFERENCE_QUERY_BOOSTS[intent]

    return sum(boosts.get(match_type, 0.0) for match_type in match_types)


def get_numbered_reference_debug(query_tokens, node):
    """Return boost and match types for numbered-reference debug output."""
    match_types = get_numbered_reference_match_types(query_tokens, node)

    return {
        "numbered_reference_boost": get_numbered_reference_boost(
            query_tokens,
            node,
        ),
        "numbered_reference_match_types": match_types,
    }


def get_numbered_reference_match_types(query_tokens, node):
    """Classify how a chunk matched a specific figure/table reference."""
    references = get_numbered_references(query_tokens)

    if not references:
        return []

    content = " ".join(node.get_content().lower().split())
    match_types = set()

    for label, number in references:
        for match in find_reference_matches(content, label, number):
            match_types.add(classify_reference_match(content, match, label))

    return sorted(match_types)


def get_numbered_reference_intent(query_tokens):
    """Classify what the user likely wants from a numbered reference."""
    token_set = set(query_tokens)

    if token_set & DISCUSSION_INTENT_TOKENS:
        return "discussion"

    if token_set & CAPTION_INTENT_TOKENS:
        return "caption"

    return "general"


def find_reference_matches(content, label, number):
    """Find all textual mentions of one figure/table number."""
    label_pattern = REFERENCE_LABEL_PATTERNS[label]

    return list(
        re.finditer(
            rf"(?<!\w){label_pattern}\s+{re.escape(number)}(?!\w)",
            content,
        )
    )


def classify_reference_match(content, match, label):
    """Classify one numbered-reference match within a chunk."""
    if is_list_entry_match(content, match, label):
        return "list_entry"

    if is_caption_match(content, match):
        return "caption"

    return "discussion_reference"


def is_list_entry_match(content, match, label):
    """Detect front-matter entries such as 'Kuva 9. ... 45'."""
    heading = LIST_SECTION_HEADINGS[label]
    heading_index = content.find(heading)
    entry_count = len(
        re.findall(
            rf"(?<!\w){REFERENCE_LABEL_PATTERNS[label]}\s+\d+\.",
            content,
        )
    )

    return heading_index != -1 and heading_index < match.start() and entry_count >= 3


def is_caption_match(content, match):
    """Detect caption-like mentions such as 'Kuva 9. Caption text'."""
    after_match = content[match.end(): match.end() + 3]

    return after_match.startswith(".")
