"""
Utility — Language Detection

Detects the programming language of a file based on its extension.
"""

_EXT_MAP = {
    "py": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "jsx": "JavaScript",
    "tsx": "TypeScript",
    "java": "Java",
    "cpp": "C++",
    "cc": "C++",
    "cxx": "C++",
    "c": "C",
    "h": "C",
    "hpp": "C++",
    "go": "Go",
    "rb": "Ruby",
    "php": "PHP",
    "cs": "C#",
    "rs": "Rust",
    "kt": "Kotlin",
    "swift": "Swift",
    "scala": "Scala",
    "sh": "Shell",
    "bash": "Shell",
    "r": "R",
    "sql": "SQL",
    "html": "HTML",
    "css": "CSS",
}

OPTIMIZED_LANGUAGE = "Python"


def detect_language(filename: str) -> str:
    """Return the programming language for *filename* based on its extension."""
    if not filename or "." not in filename:
        return "Unknown"
    ext = filename.rsplit(".", 1)[-1].lower()
    return _EXT_MAP.get(ext, "Unknown")


def detect_languages_from_chunks(chunks: list[dict]) -> dict:
    """Aggregate detected languages across all chunks.

    Returns ``{"languages": {lang: count}, "primary": str}``.
    """
    counts: dict[str, int] = {}
    for chunk in chunks:
        lang = detect_language(chunk.get("file", ""))
        counts[lang] = counts.get(lang, 0) + 1

    primary = max(counts, key=counts.get) if counts else "Unknown"
    return {"languages": counts, "primary": primary}
