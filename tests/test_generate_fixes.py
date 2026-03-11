"""
Tests for agents/nodes/generate_fixes.py

Covers:
  - Batch prompt construction
  - Batch response splitting
  - Rate-limit detection logic
  - Fallback model activation on simulated 429 errors
  - generate_fixes_node integration (with mocked LLM)
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.nodes.generate_fixes import (
    _build_batch_prompt,
    _split_batch_response,
    _is_rate_limit_error,
    _call_groq,
    generate_fixes_node,
)


# ---------------------------------------------------------------------------
# _build_batch_prompt
# ---------------------------------------------------------------------------
class TestBuildBatchPrompt:

    def test_single_issue(self):
        issues = [{"label": "Security Vulnerability", "confidence": 0.99, "code": "os.system(cmd)"}]
        prompt = _build_batch_prompt(issues)
        assert "### Issue 1" in prompt
        assert "Security Vulnerability" in prompt
        assert "os.system(cmd)" in prompt

    def test_multiple_issues(self):
        issues = [
            {"label": "Security Vulnerability", "confidence": 0.99, "code": "os.system(cmd)"},
            {"label": "Logic Flaw", "confidence": 0.85, "code": "range(len(items) + 1)"},
        ]
        prompt = _build_batch_prompt(issues)
        assert "### Issue 1" in prompt
        assert "### Issue 2" in prompt


# ---------------------------------------------------------------------------
# _split_batch_response
# ---------------------------------------------------------------------------
class TestSplitBatchResponse:

    def test_well_formatted_response(self):
        response = (
            "### Issue 1\nExplanation for issue 1.\n\n"
            "### Issue 2\nExplanation for issue 2.\n"
        )
        parts = _split_batch_response(response, 2)
        assert len(parts) == 2
        assert "Explanation for issue 1" in parts[0]
        assert "Explanation for issue 2" in parts[1]

    def test_missing_headings_returns_full_text(self):
        response = "Some unstructured response without headings."
        parts = _split_batch_response(response, 3)
        assert len(parts) == 3
        assert all(p == response for p in parts)

    def test_partial_headings(self):
        response = "### Issue 1\nFirst issue.\n### Issue 3\nThird issue.\n"
        parts = _split_batch_response(response, 3)
        assert len(parts) == 3
        assert "First issue" in parts[0]
        # Issue 2 missing — should get the full response text
        assert parts[1] == response
        assert "Third issue" in parts[2]


# ---------------------------------------------------------------------------
# _is_rate_limit_error
# ---------------------------------------------------------------------------
class TestIsRateLimitError:

    def test_429_error(self):
        err = Exception("Error code: 429 - rate limit exceeded")
        assert _is_rate_limit_error(err) is True

    def test_quota_error(self):
        err = Exception("tokens per day (TPD): Limit 100000, Used 99936")
        assert _is_rate_limit_error(err) is True

    def test_tokens_per_minute_error(self):
        err = Exception("Rate limit reached for tokens per minute")
        assert _is_rate_limit_error(err) is True

    def test_non_rate_limit_error(self):
        err = Exception("Connection timeout after 10s")
        assert _is_rate_limit_error(err) is False


# ---------------------------------------------------------------------------
# _call_groq — fallback behaviour
# ---------------------------------------------------------------------------
class TestCallGroqFallback:

    @patch("langchain_groq.ChatGroq", autospec=False)
    @patch("utils.config.require_groq_key", return_value="test-key")
    def test_fallback_on_rate_limit(self, _mock_key, mock_chatgroq_cls):
        """When the primary model raises a rate-limit error, the fallback model is used."""
        primary_llm = MagicMock()
        primary_llm.invoke.side_effect = Exception(
            "Error code: 429 - rate_limit_exceeded on tokens per day"
        )

        fallback_llm = MagicMock()
        fallback_response = MagicMock()
        fallback_response.content = "Fallback explanation."
        fallback_llm.invoke.return_value = fallback_response

        # First ChatGroq() call = primary, second = fallback
        mock_chatgroq_cls.side_effect = [primary_llm, fallback_llm]

        result = _call_groq("test prompt")

        assert result == "Fallback explanation."
        assert mock_chatgroq_cls.call_count == 2

    @patch("langchain_groq.ChatGroq", autospec=False)
    @patch("utils.config.require_groq_key", return_value="test-key")
    def test_primary_succeeds(self, _mock_key, mock_chatgroq_cls):
        """When the primary model works, no fallback is needed."""
        primary_llm = MagicMock()
        primary_response = MagicMock()
        primary_response.content = "Primary explanation."
        primary_llm.invoke.return_value = primary_response
        mock_chatgroq_cls.return_value = primary_llm

        result = _call_groq("test prompt")

        assert result == "Primary explanation."
        assert mock_chatgroq_cls.call_count == 1


# ---------------------------------------------------------------------------
# generate_fixes_node — integration (mocked LLM)
# ---------------------------------------------------------------------------
class TestGenerateFixesNode:

    @patch("agents.nodes.generate_fixes._call_groq")
    def test_batch_issues(self, mock_call):
        """All issues should be batched into a single LLM call."""
        mock_call.return_value = (
            "### Issue 1\nFix for SQL injection.\n\n"
            "### Issue 2\nFix for null ref.\n"
        )
        state = {
            "prioritized_issues": [
                {"label": "Security Vulnerability", "confidence": 0.99, "code": "sql"},
                {"label": "Null Reference Risk", "confidence": 0.85, "code": "None.x"},
            ]
        }

        result = generate_fixes_node(state)

        # Only ONE LLM call for both issues
        assert mock_call.call_count == 1
        assert len(result["fix_suggestions"]) == 2
        assert "SQL injection" in result["fix_suggestions"][0]["fix_suggestion"]
        assert "null ref" in result["fix_suggestions"][1]["fix_suggestion"]

    @patch("agents.nodes.generate_fixes._call_groq")
    def test_empty_issues(self, mock_call):
        """No issues should skip the LLM call entirely."""
        result = generate_fixes_node({"prioritized_issues": []})
        mock_call.assert_not_called()
        assert result["fix_suggestions"] == []
