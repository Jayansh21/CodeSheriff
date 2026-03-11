"""
Agent Node — generate_fixes

Calls the Groq LLM API to generate human-readable explanations and
suggested fixes for all identified issues in a **single batch prompt**.

Features:
  - Batches all issues into one LLM call to reduce token usage.
  - LLM returns structured JSON for deterministic parsing.
  - Automatic fallback from the primary model to a smaller model on
    rate-limit (429) or quota errors.
  - Retry logic with exponential backoff.
  - Graceful fallback template when JSON parsing fails.
"""

import json
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
    """Build a single prompt that asks the LLM to return structured JSON."""
    header = (
        "You are an expert code reviewer. A static analysis tool flagged the "
        "following issues in a pull request.\n\n"
        "For **each** issue, respond with a JSON array. Each element must be "
        "an object with exactly these keys:\n"
        '  - "issue_number": integer (starting from 1)\n'
        '  - "explanation": string (concise description of what is wrong)\n'
        '  - "severity": string (one of "Critical", "High", "Medium", "Low")\n'
        '  - "recommended_fix": string (brief description of how to fix it)\n'
        '  - "fixed_code": string (corrected code snippet)\n\n'
        "Return ONLY the JSON array — no markdown fences, no extra text.\n\n"
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
# Response parser — extract structured JSON from LLM output
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _fallback_entry(issue: dict) -> dict:
    """Return a safe fallback structure when JSON parsing fails for one issue."""
    return {
        "explanation": "Potential issue detected by static analysis.",
        "severity": "Medium",
        "recommended_fix": "Review this code for correctness.",
        "fixed_code": "",
    }


def _validate_entry(entry: dict) -> dict:
    """Ensure all required keys exist and are non-empty strings."""
    required_keys = ("explanation", "severity", "recommended_fix", "fixed_code")
    cleaned = {}
    for key in required_keys:
        val = entry.get(key, "")
        cleaned[key] = str(val).strip() if val else ""
    # Guarantee explanation is never blank
    if not cleaned["explanation"]:
        cleaned["explanation"] = "Potential issue detected by static analysis."
    if not cleaned["severity"]:
        cleaned["severity"] = "Medium"
    if not cleaned["recommended_fix"]:
        cleaned["recommended_fix"] = "Review this code for correctness."
    return cleaned


def _parse_json_response(response_text: str, issues: List[dict]) -> List[dict]:
    """
    Parse the LLM JSON response into a list of structured dicts.

    Returns one dict per issue with keys:
      explanation, severity, recommended_fix, fixed_code

    Falls back to a safe template when parsing fails.
    """
    # Try to extract a JSON array even if the LLM wrapped it in markdown fences
    cleaned = response_text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # Find the JSON array in the response
    match = _JSON_ARRAY_RE.search(cleaned)
    if not match:
        logger.warning("No JSON array found in LLM response — using fallback.")
        return [_fallback_entry(issue) for issue in issues]

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.warning("JSON decode failed: %s — using fallback.", e)
        return [_fallback_entry(issue) for issue in issues]

    if not isinstance(parsed, list):
        logger.warning("LLM returned non-array JSON — using fallback.")
        return [_fallback_entry(issue) for issue in issues]

    # Map by issue_number, default to order
    by_number: dict[int, dict] = {}
    for i, item in enumerate(parsed):
        if isinstance(item, dict):
            num = item.get("issue_number", i + 1)
            by_number[int(num)] = item

    results = []
    for idx, issue in enumerate(issues, 1):
        entry = by_number.get(idx)
        if entry:
            results.append(_validate_entry(entry))
        else:
            results.append(_fallback_entry(issue))

    return results


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
    structured = _parse_json_response(response_text, issues)

    suggestions = []
    for issue, fields in zip(issues, structured):
        suggestions.append({
            **issue,
            "explanation": fields["explanation"],
            "severity": fields["severity"],
            "recommended_fix": fields["recommended_fix"],
            "fixed_code": fields["fixed_code"],
            # Keep a backward-compatible key
            "fix_suggestion": fields["explanation"],
        })

    logger.info("Generated %d fix suggestion(s).", len(suggestions))
    return {"fix_suggestions": suggestions}
