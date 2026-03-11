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


def _parse_fix_sections(fix_text: str) -> dict:
    """
    Try to split LLM fix text into explanation, impact, and suggested fix.
    Returns the full text for any section that can't be isolated.
    """
    explanation = fix_text
    impact = ""
    suggested_fix = ""

    # Try to extract a code block as the suggested fix
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", fix_text, re.DOTALL)
    if code_match:
        suggested_fix = code_match.group(1).strip()

    # Try to split on common LLM headings
    severity_match = re.search(
        r"\*?\*?Severity\*?\*?[:\s]*(Critical|High|Medium|Low)",
        fix_text, re.IGNORECASE,
    )

    # Everything before severity / code block is the explanation
    split_pos = len(fix_text)
    if severity_match:
        split_pos = min(split_pos, severity_match.start())
    if code_match:
        # Look backwards for a heading before the code block
        pre_code = fix_text[: code_match.start()]
        heading_match = re.search(r"\*?\*?Suggested [Ff]ix\*?\*?", pre_code)
        if heading_match:
            split_pos = min(split_pos, heading_match.start())
        else:
            split_pos = min(split_pos, code_match.start())

    explanation = fix_text[:split_pos].strip()
    if severity_match:
        impact = f"Severity: **{severity_match.group(1)}**"

    return {
        "explanation": explanation or fix_text,
        "impact": impact,
        "suggested_fix": suggested_fix,
    }


def _build_inline_body(label: str, confidence: float, fix_text: str) -> str:
    """
    Build a Copilot-style inline comment body.
    Preserves the FULL fix text — no truncation.
    """
    sections = _parse_fix_sections(fix_text)

    parts = [
        f"**\U0001f6e1\ufe0f CodeSheriff \u2014 {label} (Confidence: {confidence:.0%})**\n",
        "**Issue**\n",
        f"{sections['explanation']}\n",
    ]

    if sections["impact"]:
        parts.append("**Why this matters**\n")
        parts.append(f"{sections['impact']}\n")

    if sections["suggested_fix"]:
        parts.append("**Recommended fix**\n")
        parts.append(f"```python\n{sections['suggested_fix']}\n```\n")

    return "\n".join(parts)


def format_review_node(state: dict) -> dict:
    """LangGraph node: produce a final Markdown review string + inline metadata."""
    suggestions: List[dict] = state.get("fix_suggestions", [])

    if not suggestions:
        review = (
            "# \U0001f6e1\ufe0f CodeSheriff Review\n\n"
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
        fix = s.get("fix_suggestion", "N/A")
        file_path = s.get("file", "unknown")
        start_line = s.get("start_line", 0)

        type_counter[label] += 1

        # --- Full review section (no truncation) ---
        lines.append(f"## Issue {idx}: {label}\n")
        lines.append(f"**Confidence:** {conf:.0%}\n")
        lines.append(f"**File:** `{file_path}`\n")
        lines.append("**Problematic code:**\n")
        lines.append(f"```python\n{code}\n```\n")
        lines.append("**Analysis & Suggested Fix:**\n")
        lines.append(f"{fix}\n")
        lines.append("---\n")

        # --- Determine the best line number for the inline comment ---
        issue_offset = _find_issue_line_offset(code)
        best_line = max((start_line or 1) + issue_offset, 1)

        # --- Build Copilot-style inline body (full text, no slicing) ---
        inline_body = _build_inline_body(label, conf, fix)

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
