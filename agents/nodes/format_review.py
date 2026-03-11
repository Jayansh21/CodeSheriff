"""
Agent Node — format_review

Assembles all fix suggestions into:
  1. A compact **summary** with grouped issue counts (status comment).
  2. Per-issue **inline_comments** in Copilot-style markdown.
  3. The full Markdown **final_review** (backward-compatible).
"""

import re
import sys
from collections import Counter
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger
from utils.language_detection import OPTIMIZED_LANGUAGE

logger = get_logger("agents.nodes.format_review")

# Regex patterns used to try to locate the most relevant line within a chunk
_ISSUE_LINE_PATTERNS = [
    re.compile(r"""['"]SELECT\s.*?\+""", re.IGNORECASE),
    re.compile(r"""\beval\s*\("""),
    re.compile(r"""\bos\.system\s*\("""),
    re.compile(r"""\bexec\s*\("""),
    re.compile(r"""\bsubprocess\.\w+\([^)]*shell\s*=\s*True"""),
    re.compile(r"""\.fetchone\(\)\.\w+"""),
    re.compile(r"""=\s*None"""),
    re.compile(r"""/\s*0\b"""),
    re.compile(r"""\bif\s+\w+\s*=[^=]"""),
    re.compile(r"""range\(len\("""),
]


def _find_issue_line_offset(code: str) -> int:
    """Return 0-based line offset of the most suspicious line in *code*."""
    for pattern in _ISSUE_LINE_PATTERNS:
        for i, line in enumerate(code.splitlines()):
            if pattern.search(line):
                return i
    return 0


def _build_inline_body(label: str, confidence: float, suggestion: dict) -> str:
    """
    Build a Copilot-style inline comment body from structured fields.

    Uses the structured keys: explanation, severity, recommended_fix, fixed_code.
    Falls back to fix_suggestion (raw text) if structured keys are missing.
    """
    explanation = suggestion.get("explanation", "").strip()
    severity = suggestion.get("severity", "").strip()
    recommended_fix = suggestion.get("recommended_fix", "").strip()
    fixed_code = suggestion.get("fixed_code", "").strip()

    # Fallback: if no structured explanation, use raw fix_suggestion
    if not explanation:
        explanation = suggestion.get("fix_suggestion", "No details available.").strip()

    parts = [
        f"**\U0001f6e1\ufe0f CodeSheriff \u2014 {label} (Confidence: {confidence:.0%})**\n",
        "**Issue**\n",
        f"{explanation}\n",
    ]

    if severity:
        parts.append(f"**Severity:** {severity}\n")

    if recommended_fix:
        parts.append("**Recommended fix**\n")
        parts.append(f"{recommended_fix}\n")

    if fixed_code:
        parts.append("**Fixed code**\n")
        parts.append(f"```python\n{fixed_code}\n```\n")

    return "\n".join(parts)


def format_review_node(state: dict) -> dict:
    """LangGraph node: produce a final Markdown review string + inline metadata."""
    suggestions: List[dict] = state.get("fix_suggestions", [])
    language = state.get("language", "Unknown")

    # Build the language header
    lang_header = f"**Language detected:** {language}\n"
    if language != OPTIMIZED_LANGUAGE and language != "Unknown":
        lang_header += f"\n\u26a0\ufe0f _CodeSheriff is currently optimized for {OPTIMIZED_LANGUAGE} analysis._\n"

    if not suggestions:
        review = (
            "# \U0001f6e1\ufe0f CodeSheriff Review\n\n"
            f"{lang_header}\n"
            "No issues detected \u2014 the code looks clean! :white_check_mark:\n"
        )
        return {
            "final_review": review,
            "review_summary": review,
            "inline_comments": [],
        }

    # ---- Full review (backward-compatible) ----
    lines = [
        "# \U0001f6e1\ufe0f CodeSheriff Review\n",
        f"{lang_header}\n",
        f"**Issues found:** {len(suggestions)}\n",
        "---\n",
    ]

    # ---- Inline comment metadata ----
    inline_comments: List[dict] = []

    # ---- Count issue types for grouped summary ----
    type_counter: Counter = Counter()

    for idx, s in enumerate(suggestions, 1):
        label = s.get("label", "Unknown")
        conf = s.get("confidence", 0)
        code = s.get("code", "")
        file_path = s.get("file", "unknown")
        start_line = s.get("start_line", 0)

        explanation = s.get("explanation", "").strip()
        severity = s.get("severity", "").strip()
        recommended_fix = s.get("recommended_fix", "").strip()
        fixed_code = s.get("fixed_code", "").strip()

        # Fallback: use raw fix_suggestion if structured fields are absent
        if not explanation:
            explanation = s.get("fix_suggestion", "N/A").strip()

        type_counter[label] += 1

        # --- Full review section ---
        lines.append(f"## Issue {idx}: {label}\n")
        lines.append(f"**Confidence:** {conf:.0%}\n")
        lines.append(f"**File:** `{file_path}`\n")
        if severity:
            lines.append(f"**Severity:** {severity}\n")
        lines.append("**Problematic code:**\n")
        lines.append(f"```python\n{code}\n```\n")
        lines.append("**Explanation:**\n")
        lines.append(f"{explanation}\n")
        if recommended_fix:
            lines.append("**Recommended fix:**\n")
            lines.append(f"{recommended_fix}\n")
        if fixed_code:
            lines.append("**Fixed code:**\n")
            lines.append(f"```python\n{fixed_code}\n```\n")
        lines.append("---\n")

        # --- Determine the best line number for the inline comment ---
        issue_offset = _find_issue_line_offset(code)
        best_line = max((start_line or 1) + issue_offset, 1)

        # --- Build Copilot-style inline body (structured fields) ---
        inline_body = _build_inline_body(label, conf, s)

        inline_comments.append({
            "file": file_path,
            "line": best_line,
            "label": label,
            "confidence": conf,
            "body": inline_body,
        })

    review = "\n".join(lines)

    # ---- Grouped summary for the status comment ----
    total = sum(type_counter.values())
    summary_lines = [
        "# \U0001f6e1\ufe0f CodeSheriff Review\n",
        f"{lang_header}\n",
        f"**{total} issue{'s' if total != 1 else ''} detected.**\n",
    ]
    for issue_type, count in type_counter.most_common():
        plural = "ies" if issue_type.endswith("y") and count > 1 else "s" if count > 1 else ""
        display_type = issue_type.rstrip("y") + plural if issue_type.endswith("y") and count > 1 else (issue_type + plural if count > 1 else issue_type)
        summary_lines.append(f"- **{display_type}:** {count}")
    summary_lines.append("\n_Inline comments have been added to the affected lines._")
    review_summary = "\n".join(summary_lines)

    logger.info("Final review formatted (%d issues, %d inline comments).", len(suggestions), len(inline_comments))
    return {
        "final_review": review,
        "review_summary": review_summary,
        "inline_comments": inline_comments,
    }
