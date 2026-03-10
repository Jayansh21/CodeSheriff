"""
Agent Node — generate_fixes

Calls the Groq LLM API to generate human-readable
explanations and suggested fixes for each identified issue.

Includes retry logic (up to 3 attempts) and timeout handling.
"""

import sys
import time
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import (
    GROQ_MODEL_NAME,
    GROQ_MAX_RETRIES,
    GROQ_TIMEOUT_SECONDS,
)
from utils.logger import get_logger

logger = get_logger("agents.nodes.generate_fixes")


def _build_prompt(issue: dict) -> str:
    """Build the LLM prompt for a single issue."""
    return (
        "You are an expert code reviewer. A static analysis tool flagged the "
        "following code snippet as potentially buggy.\n\n"
        f"**Bug type detected:** {issue.get('label', 'Unknown')}\n"
        f"**Confidence:** {issue.get('confidence', 0):.0%}\n\n"
        "```python\n"
        f"{issue.get('code', '')}\n"
        "```\n\n"
        "Please provide:\n"
        "1. A concise **explanation** of what is wrong.\n"
        "2. A **severity** rating (Critical / High / Medium / Low).\n"
        "3. A **suggested fix** as a corrected code snippet.\n"
    )


def _call_groq(prompt: str) -> str:
    """
    Call the Groq API with retry and timeout.
    Returns the LLM response text, or a fallback message on failure.
    """
    try:
        from utils.config import require_groq_key
        api_key = require_groq_key()
    except EnvironmentError as e:
        logger.warning(f"Groq API key not set: {e}")
        return "[LLM unavailable — Groq API key not configured]"

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        logger.warning("langchain-groq not installed.")
        return "[LLM unavailable — langchain-groq not installed]"

    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
        timeout=GROQ_TIMEOUT_SECONDS,
    )

    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.warning(f"Groq API attempt {attempt}/{GROQ_MAX_RETRIES} failed: {e}")
            if attempt < GROQ_MAX_RETRIES:
                time.sleep(2 ** attempt)  # exponential backoff

    return "[LLM unavailable — all retry attempts exhausted]"


def generate_fixes_node(state: dict) -> dict:
    """LangGraph node: generate LLM-powered fix suggestions for each issue."""
    issues: List[dict] = state.get("prioritized_issues", [])
    suggestions = []

    for i, issue in enumerate(issues):
        logger.info(
            f"Generating fix for issue {i + 1}/{len(issues)}: "
            f"{issue.get('label', '?')} …"
        )
        prompt = _build_prompt(issue)
        explanation = _call_groq(prompt)

        suggestions.append({
            **issue,
            "fix_suggestion": explanation,
        })

    logger.info(f"Generated {len(suggestions)} fix suggestion(s).")
    return {"fix_suggestions": suggestions}
