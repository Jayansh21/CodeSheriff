"""
Agent Node — generate_fixes

Calls the Groq LLM API to generate human-readable explanations and
suggested fixes for all identified issues in a **single batch prompt**.

Features:
  - Batches all issues into one LLM call to reduce token usage.
  - Automatic fallback from the primary model to a smaller model on
    rate-limit (429) or quota errors.
  - Retry logic with exponential backoff.
"""

import re
import sys
import time
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import (
    GROQ_MODEL_NAME,
    GROQ_FALLBACK_MODEL,
    GROQ_MAX_RETRIES,
    GROQ_TIMEOUT_SECONDS,
)
from utils.logger import get_logger

logger = get_logger("agents.nodes.generate_fixes")

# Errors that indicate a rate-limit / quota problem → trigger fallback
_RATE_LIMIT_MARKERS = ("rate_limit", "429", "tokens per day", "tokens per minute", "quota")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a Groq rate-limit or quota error."""
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


# ---------------------------------------------------------------------------
# Batch prompt builder
# ---------------------------------------------------------------------------

def _build_batch_prompt(issues: List[dict]) -> str:
    """Build a single prompt that asks the LLM to analyse all issues at once."""
    header = (
        "You are an expert code reviewer. A static analysis tool flagged the "
        "following issues in a pull request.\n\n"
        "For **each** issue, provide:\n"
        "1. A concise **explanation** of what is wrong.\n"
        "2. A **severity** rating (Critical / High / Medium / Low).\n"
        "3. A **suggested fix** as a corrected code snippet.\n\n"
        "Separate your answers clearly using the exact heading format:\n"
        "### Issue N\n"
        "(where N is the issue number)\n\n"
        "---\n\n"
    )

    parts: list[str] = [header]
    for idx, issue in enumerate(issues, 1):
        parts.append(
            f"### Issue {idx}\n"
            f"**Bug type detected:** {issue.get('label', 'Unknown')}\n"
            f"**Confidence:** {issue.get('confidence', 0):.0%}\n\n"
            "```python\n"
            f"{issue.get('code', '')}\n"
            "```\n\n"
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Groq LLM call with fallback
# ---------------------------------------------------------------------------

def _call_groq(prompt: str) -> str:
    """
    Call the Groq API.  Tries the primary model first; on rate-limit /
    quota errors, transparently falls back to the smaller model.

    Returns the LLM response text, or a placeholder on failure.
    """
    try:
        from utils.config import require_groq_key
        api_key = require_groq_key()
    except EnvironmentError as e:
        logger.warning("Groq API key not set: %s", e)
        return "[LLM unavailable — Groq API key not configured]"

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        logger.warning("langchain-groq not installed.")
        return "[LLM unavailable — langchain-groq not installed]"

    models_to_try = [GROQ_MODEL_NAME, GROQ_FALLBACK_MODEL]

    for model_name in models_to_try:
        logger.info("Using Groq model: %s", model_name)

        llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.3,
            max_tokens=4096,
            timeout=GROQ_TIMEOUT_SECONDS,
        )

        for attempt in range(1, GROQ_MAX_RETRIES + 1):
            try:
                response = llm.invoke(prompt)
                return response.content
            except Exception as e:
                logger.warning(
                    "Groq API attempt %d/%d failed (model=%s): %s",
                    attempt, GROQ_MAX_RETRIES, model_name, e,
                )
                if _is_rate_limit_error(e):
                    # Skip remaining retries for this model — jump to fallback
                    logger.info("Rate limit hit on %s — falling back to next model.", model_name)
                    break
                if attempt < GROQ_MAX_RETRIES:
                    time.sleep(2 ** attempt)
        else:
            # All retries exhausted for this model (non-rate-limit error)
            continue
        # Broke out of retry loop (rate limit) — try next model
        continue

    return "[LLM unavailable — all retry attempts exhausted]"


# ---------------------------------------------------------------------------
# Response parser — split batch LLM output back into per-issue text
# ---------------------------------------------------------------------------

_ISSUE_HEADING_RE = re.compile(r"###\s*Issue\s+(\d+)", re.IGNORECASE)


def _split_batch_response(response_text: str, issue_count: int) -> List[str]:
    """Split a batch LLM response into per-issue explanation strings."""
    # Find all "### Issue N" headings
    splits = list(_ISSUE_HEADING_RE.finditer(response_text))

    if not splits:
        # LLM didn't follow heading format — give the full text to every issue
        return [response_text] * issue_count

    parts: dict[int, str] = {}
    for i, match in enumerate(splits):
        issue_num = int(match.group(1))
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(response_text)
        parts[issue_num] = response_text[start:end].strip()

    return [parts.get(idx, response_text) for idx in range(1, issue_count + 1)]


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def generate_fixes_node(state: dict) -> dict:
    """LangGraph node: generate LLM-powered fix suggestions for all issues."""
    issues: List[dict] = state.get("prioritized_issues", [])

    if not issues:
        logger.info("No issues to generate fixes for.")
        return {"fix_suggestions": []}

    logger.info(
        "Generating explanations for %d issue(s) in a single batch prompt.", len(issues)
    )

    prompt = _build_batch_prompt(issues)
    response_text = _call_groq(prompt)
    explanations = _split_batch_response(response_text, len(issues))

    suggestions = []
    for issue, explanation in zip(issues, explanations):
        suggestions.append({**issue, "fix_suggestion": explanation})

    logger.info("Generated %d fix suggestion(s).", len(suggestions))
    return {"fix_suggestions": suggestions}
