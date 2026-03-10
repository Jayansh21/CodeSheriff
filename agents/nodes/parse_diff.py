"""
Agent Node — parse_diff

Splits a unified diff string into individual code-change chunks.
Each chunk is the body of a hunk (the changed lines with +/- prefixes removed).
"""

import re
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("agents.nodes.parse_diff")


# Number of context lines (unchanged lines from the diff) to include
# before and after each group of removed lines.
_CONTEXT_LINES = 3


def _extract_chunks(diff_text: str) -> List[str]:
    """
    Parse a unified diff and return a list of meaningful code chunks.

    Strategy: collect contiguous groups of added lines ('+' prefix),
    stripping the prefix.  Lines starting with '+++' (file header) are
    skipped.  Each contiguous block of additions becomes one chunk.
    """
    if not diff_text or not diff_text.strip():
        logger.warning("Empty diff received.")
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []

    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            current_chunk.append(line[1:])
        else:
            if current_chunk:
                text = "\n".join(current_chunk)
                if text.strip():
                    chunks.append(text)
                current_chunk = []

    # Flush last chunk
    if current_chunk:
        text = "\n".join(current_chunk)
        if text.strip():
            chunks.append(text)

    logger.info("Parsed %d code chunk(s) from diff.", len(chunks))
    return chunks


def parse_diff_node(state: dict) -> dict:
    """LangGraph node: parse the PR diff into code chunks."""
    diff = state.get("pr_diff", "")
    chunks = _extract_chunks(diff)
    return {"code_chunks": chunks}
