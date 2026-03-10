"""
Agent Node — format_review

Assembles all fix suggestions into:
  1. A compact **summary** (used to update the status comment).
  2. Per-issue **inline_comments** metadata for line-level PR comments.
  3. The full Markdown **final_review** (backward-compatible).
"""

import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("agents.nodes.format_review")


def format_review_node(state: dict) -> dict:
    """LangGraph node: produce a final Markdown review string + inline metadata."""
    suggestions: List[dict] = state.get("fix_suggestions", [])

    if not suggestions:
        review = (
            "# \U0001f6e1\ufe0f CodeSheriff Review\n\n"
            "No issues detected — the code looks clean! :white_check_mark:\n"
        )
        return {
            "final_review": review,
            "review_summary": review,
            "inline_comments": [],
        }

    # ---- Full review (backward-compatible) ----
    lines = [
        "# \U0001f6e1\ufe0f CodeSheriff Review\n",
        f"**Issues found:** {len(suggestions)}\n",
        "---\n",
    ]

    # ---- Inline comment metadata ----
    inline_comments: List[dict] = []

    # ---- Summary bullets ----
    summary_bullets: List[str] = []

    for idx, s in enumerate(suggestions, 1):
        label = s.get("label", "Unknown")
        conf = s.get("confidence", 0)
        code = s.get("code", "")
        fix = s.get("fix_suggestion", "N/A")
        file_path = s.get("file", "unknown")
        start_line = s.get("start_line", 0)

        lines.append(f"## Issue {idx}: {label}\n")
        lines.append(f"**Confidence:** {conf:.0%}\n")
        lines.append(f"**File:** `{file_path}`\n")
        lines.append("**Problematic code:**\n")
        lines.append(f"```python\n{code}\n```\n")
        lines.append("**Analysis & Suggested Fix:**\n")
        lines.append(f"{fix}\n")
        lines.append("---\n")

        summary_bullets.append(f"- **{label}** — {fix[:80]}{'…' if len(fix) > 80 else ''}")

        inline_comments.append({
            "file": file_path,
            "line": start_line,
            "label": label,
            "confidence": conf,
            "body": f"**\U0001f6e1\ufe0f CodeSheriff — {label}** (confidence: {conf:.0%})\n\n{fix[:500]}",
        })

    review = "\n".join(lines)

    # ---- Compact summary for the status comment ----
    summary_lines = [
        "# \U0001f6e1\ufe0f CodeSheriff Review\n",
        f"**Issues detected:** {len(suggestions)}\n",
    ]
    summary_lines.extend(summary_bullets)
    summary_lines.append("\n_Inline comments have been added to the affected lines._")
    review_summary = "\n".join(summary_lines)

    logger.info("Final review formatted (%d issues, %d inline comments).", len(suggestions), len(inline_comments))
    return {
        "final_review": review,
        "review_summary": review_summary,
        "inline_comments": inline_comments,
    }
