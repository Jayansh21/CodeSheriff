"""
Agent Node — format_review

Assembles all fix suggestions into a single Markdown-formatted review.
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
    """LangGraph node: produce a final Markdown review string."""
    suggestions: List[dict] = state.get("fix_suggestions", [])

    if not suggestions:
        review = (
            "# CodeSheriff Review\n\n"
            "No issues detected — the code looks clean! :white_check_mark:\n"
        )
        return {"final_review": review}

    lines = [
        "# CodeSheriff Review\n",
        f"**Issues found:** {len(suggestions)}\n",
        "---\n",
    ]

    for idx, s in enumerate(suggestions, 1):
        lines.append(f"## Issue {idx}: {s.get('label', 'Unknown')}\n")
        lines.append(f"**Confidence:** {s.get('confidence', 0):.0%}\n")
        lines.append("**Problematic code:**\n")
        lines.append(f"```python\n{s.get('code', '')}\n```\n")
        lines.append("**Analysis & Suggested Fix:**\n")
        lines.append(f"{s.get('fix_suggestion', 'N/A')}\n")
        lines.append("---\n")

    review = "\n".join(lines)
    logger.info("Final review formatted.")
    return {"final_review": review}
