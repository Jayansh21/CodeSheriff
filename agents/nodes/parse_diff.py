"""
Agent Node — parse_diff

Splits a unified diff string into *function-level* code chunks.

Strategy:
  1. Collect added lines (``+`` prefix, excluding ``+++`` file headers)
     grouped by file.
  2. Within each file, split at function / class boundaries so that
     separate definitions become separate chunks.
  3. Cap each chunk at ``_MAX_CHUNK_LINES`` lines.

Each chunk dict carries ``code`` (str), ``file`` (str) and ``start_line`` (int)
so that downstream nodes can post inline PR comments.
"""

import re
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger
from utils.language_detection import detect_languages_from_chunks

logger = get_logger("agents.nodes.parse_diff")

_MAX_CHUNK_LINES = 40
_FUNC_RE = re.compile(r"^\s*(?:def |class |async def )\w+")


def _extract_chunks(diff_text: str) -> List[dict]:
    """Return a list of chunk dicts ``{code, file, start_line}``."""
    if not diff_text or not diff_text.strip():
        logger.warning("Empty diff received.")
        return []

    # ------------------------------------------------------------------
    # Pass 1 — group added lines by file, recording diff line numbers
    # ------------------------------------------------------------------
    current_file = "unknown"
    diff_line_no = 0            # 1-based line counter in the new file
    file_groups: dict = {}      # file -> [(code_line, diff_line_no)]

    for raw in diff_text.splitlines():
        if raw.startswith("+++ b/"):
            current_file = raw[6:]
            continue
        if raw.startswith("+++"):
            continue

        # Hunk header — extract starting line number in new file
        hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)", raw)
        if hunk_match:
            diff_line_no = int(hunk_match.group(1))
            continue

        if raw.startswith("-"):
            # Deleted line — doesn't exist in the new file
            continue

        if raw.startswith("+"):
            code_line = raw[1:]
            file_groups.setdefault(current_file, []).append(
                (code_line, diff_line_no)
            )
            diff_line_no += 1
            continue

        # Context line — count it in the new-file numbering
        diff_line_no += 1

    # ------------------------------------------------------------------
    # Pass 2 — split each file's added lines at function/class boundaries
    # ------------------------------------------------------------------
    chunks: List[dict] = []

    for filepath, lines_info in file_groups.items():
        current_lines: List[str] = []
        chunk_start = lines_info[0][1] if lines_info else 1

        for code_line, line_no in lines_info:
            is_boundary = _FUNC_RE.match(code_line)

            if is_boundary and current_lines:
                # Flush previous chunk
                text = "\n".join(current_lines)
                if text.strip():
                    chunks.append({"code": text, "file": filepath, "start_line": chunk_start})
                current_lines = []
                chunk_start = line_no

            current_lines.append(code_line)

            # Cap chunk size
            if len(current_lines) >= _MAX_CHUNK_LINES:
                text = "\n".join(current_lines)
                if text.strip():
                    chunks.append({"code": text, "file": filepath, "start_line": chunk_start})
                current_lines = []
                chunk_start = line_no + 1

        # Flush remaining lines
        if current_lines:
            text = "\n".join(current_lines)
            if text.strip():
                chunks.append({"code": text, "file": filepath, "start_line": chunk_start})

    logger.info("Parsed %d code chunk(s) from diff.", len(chunks))
    return chunks


def parse_diff_node(state: dict) -> dict:
    """LangGraph node: parse the PR diff into code chunks."""
    diff = state.get("pr_diff", "")
    chunks = _extract_chunks(diff)
    lang_info = detect_languages_from_chunks(chunks)
    return {
        "code_chunks": chunks,
        "language": lang_info["primary"],
        "languages": lang_info["languages"],
    }
