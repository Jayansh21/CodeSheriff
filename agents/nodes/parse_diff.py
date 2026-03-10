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

    Strategy:
    1. Split on hunk headers (@@).
    2. For each hunk, identify groups of consecutive removed lines ('-').
    3. For each group, include up to _CONTEXT_LINES surrounding context
       lines (unchanged lines starting with ' ') so the classifier can
       see the function/variable context around the buggy code.
    """
    if not diff_text or not diff_text.strip():
        logger.warning("Empty diff received.")
        return []

    chunks: List[str] = []
    hunks = re.split(r"^@@.*?@@.*$", diff_text, flags=re.MULTILINE)

    for hunk in hunks:
        lines = hunk.splitlines()
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith("-") and not stripped.startswith("---"):
                # Found start of a removed-line group
                group_start = i
                removed: List[str] = []
                while i < len(lines):
                    s = lines[i].strip()
                    if s.startswith("-") and not s.startswith("---"):
                        code_line = lines[i].lstrip("-").rstrip()
                        if code_line.strip():
                            removed.append(code_line)
                        i += 1
                    else:
                        break
                group_end = i

                if not removed:
                    continue

                # Context lines before the group (skip +/- lines)
                before: List[str] = []
                for j in range(max(0, group_start - _CONTEXT_LINES), group_start):
                    ln = lines[j]
                    if ln.startswith(" ") or (ln and not ln[0] in "+-"):
                        before.append(ln[1:].rstrip() if ln.startswith(" ") else ln.rstrip())

                # Context lines after the group (skip +/- lines)
                after: List[str] = []
                count = 0
                for j in range(group_end, len(lines)):
                    if count >= _CONTEXT_LINES:
                        break
                    ln = lines[j]
                    if ln.startswith(" ") or (ln and not ln[0] in "+-"):
                        after.append(ln[1:].rstrip() if ln.startswith(" ") else ln.rstrip())
                        count += 1
                    elif ln.strip().startswith("+"):
                        continue  # skip added lines
                    else:
                        break

                chunk = "\n".join(before + removed + after)
                if chunk.strip():
                    chunks.append(chunk)
            else:
                i += 1

    logger.info(f"Parsed {len(chunks)} code chunk(s) from diff.")
    return chunks


def parse_diff_node(state: dict) -> dict:
    """LangGraph node: parse the PR diff into code chunks."""
    diff = state.get("pr_diff", "")
    chunks = _extract_chunks(diff)
    return {"code_chunks": chunks}
