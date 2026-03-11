"""
Tests for agents/nodes/generate_fixes.py

Covers:
  - Batch prompt construction (JSON output request)
  - JSON response parsing + validation + fallback
  - Rate-limit detection logic
  - Fallback model activation on simulated 429 errors
  - generate_fixes_node integration (with mocked LLM)
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.nodes.generate_fixes import (
    _build_batch_prompt,
    _parse_json_response,
    _validate_entry,
    _fallback_entry,
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
        assert "Issue 1" in prompt
        assert "Security Vulnerability" in prompt
        assert "os.system(cmd)" in prompt
        assert "JSON array" in prompt

    def test_multiple_issues(self):
        issues = [
            {"label": "Security Vulnerability", "confidence": 0.99, "code": "os.system(cmd)"},
            {"label": "Logic Flaw", "confidence": 0.85, "code": "range(len(items) + 1)"},
        ]
        prompt = _build_batch_prompt(issues)
        assert "Issue 1" in prompt
        assert "Issue 2" in prompt

    def test_prompt_requests_json_keys(self):
        issues = [{"label": "Bug", "confidence": 0.9, "code": "x=1"}]
        prompt = _build_batch_prompt(issues)
        assert "explanation" in prompt
        assert "severity" in prompt
        assert "recommended_fix" in prompt
        assert "fixed_code" in prompt


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------
class TestParseJsonResponse:

    def test_valid_json(self):
        response = json.dumps([
            {
                "issue_number": 1,
                "explanation": "SQL injection risk.",
                "severity": "Critical",
                "recommended_fix": "Use parameterized queries.",
                "fixed_code": "cursor.execute(sql, (param,))",
            }
        ])
        issues = [{"label": "Security Vulnerability", "confidence": 0.99, "code": "sql"}]
        result = _parse_json_response(response, issues)
        assert len(result) == 1
        assert result[0]["explanation"] == "SQL injection risk."
        assert result[0]["severity"] == "Critical"
        assert result[0]["recommended_fix"] == "Use parameterized queries."
        assert result[0]["fixed_code"] == "cursor.execute(sql, (param,))"

    def test_json_with_markdown_fences(self):
        inner = json.dumps([{
            "issue_number": 1,
            "explanation": "Null ref risk.",
            "severity": "High",
            "recommended_fix": "Check for None.",
            "fixed_code": "if x is not None:",
        }])
        response = f"```json\n{inner}\n```"
        issues = [{"label": "Null Ref", "confidence": 0.9, "code": "x.y"}]
        result = _parse_json_response(response, issues)
        assert result[0]["explanation"] == "Null ref risk."

    def test_invalid_json_returns_fallback(self):
        response = "This is not JSON at all."
        issues = [{"label": "Bug", "confidence": 0.8, "code": "x"}]
        result = _parse_json_response(response, issues)
        assert len(result) == 1
        assert result[0]["explanation"] == "Potential issue detected by static analysis."
        assert result[0]["severity"] == "Medium"

    def test_partial_json_returns_fallback_for_missing(self):
        response = json.dumps([
            {
                "issue_number": 1,
                "explanation": "First issue.",
                "severity": "Low",
                "recommended_fix": "Fix it.",
                "fixed_code": "",
            }
        ])
        issues = [
            {"label": "Bug1", "confidence": 0.9, "code": "a"},
            {"label": "Bug2", "confidence": 0.8, "code": "b"},
        ]
        result = _parse_json_response(response, issues)
        assert len(result) == 2
        assert result[0]["explanation"] == "First issue."
        # Issue 2 missing from LLM response — fallback
        assert result[1]["explanation"] == "Potential issue detected by static analysis."

    def test_multiple_issues_json(self):
        response = json.dumps([
            {
                "issue_number": 1,
                "explanation": "SQL injection.",
                "severity": "Critical",
                "recommended_fix": "Parameterize.",
                "fixed_code": "cursor.execute(sql, params)",
            },
            {
                "issue_number": 2,
                "explanation": "Null deref.",
                "severity": "High",
                "recommended_fix": "Add null check.",
                "fixed_code": "if obj: obj.method()",
            },
        ])
        issues = [
            {"label": "Sec", "confidence": 0.99, "code": "sql"},
            {"label": "Null", "confidence": 0.85, "code": "obj.x"},
        ]
        result = _parse_json_response(response, issues)
        assert len(result) == 2
        assert result[0]["explanation"] == "SQL injection."
        assert result[1]["explanation"] == "Null deref."

    def test_non_array_json_returns_fallback(self):
        response = json.dumps({"error": "unexpected format"})
        issues = [{"label": "Bug", "confidence": 0.5, "code": "x"}]
        result = _parse_json_response(response, issues)
        assert result[0]["severity"] == "Medium"


# ---------------------------------------------------------------------------
# _validate_entry / _fallback_entry
# ---------------------------------------------------------------------------
class TestValidation:

    def test_validate_fills_missing_keys(self):
        entry = {"explanation": "Good", "severity": ""}
        result = _validate_entry(entry)
        assert result["explanation"] == "Good"
        assert result["severity"] == "Medium"  # default
        assert result["recommended_fix"] == "Review this code for correctness."

    def test_validate_strips_whitespace(self):
        entry = {
            "explanation": "  Foo  ",
            "severity": " High ",
            "recommended_fix": " Use X ",
            "fixed_code": "  code  ",
        }
        result = _validate_entry(entry)
        assert result["explanation"] == "Foo"
        assert result["severity"] == "High"

    def test_fallback_entry_has_all_keys(self):
        fb = _fallback_entry({"label": "Bug"})
        assert "explanation" in fb
        assert "severity" in fb
        assert "recommended_fix" in fb
        assert "fixed_code" in fb
        assert fb["explanation"] != ""


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
    def test_batch_issues_json(self, mock_call):
        """All issues should be batched into a single LLM call with JSON response."""
        mock_call.return_value = json.dumps([
            {
                "issue_number": 1,
                "explanation": "Fix for SQL injection.",
                "severity": "Critical",
                "recommended_fix": "Use parameterized queries.",
                "fixed_code": "cursor.execute(sql, (p,))",
            },
            {
                "issue_number": 2,
                "explanation": "Fix for null ref.",
                "severity": "High",
                "recommended_fix": "Add None check.",
                "fixed_code": "if x is not None: x.y",
            },
        ])
        state = {
            "prioritized_issues": [
                {"label": "Security Vulnerability", "confidence": 0.99, "code": "sql"},
                {"label": "Null Reference Risk", "confidence": 0.85, "code": "None.x"},
            ]
        }

        result = generate_fixes_node(state)

        assert mock_call.call_count == 1
        assert len(result["fix_suggestions"]) == 2
        s0 = result["fix_suggestions"][0]
        assert s0["explanation"] == "Fix for SQL injection."
        assert s0["severity"] == "Critical"
        assert s0["recommended_fix"] == "Use parameterized queries."
        assert s0["fixed_code"] == "cursor.execute(sql, (p,))"
        # backward-compat key
        assert s0["fix_suggestion"] == "Fix for SQL injection."

        s1 = result["fix_suggestions"][1]
        assert s1["explanation"] == "Fix for null ref."

    @patch("agents.nodes.generate_fixes._call_groq")
    def test_empty_issues(self, mock_call):
        """No issues should skip the LLM call entirely."""
        result = generate_fixes_node({"prioritized_issues": []})
        mock_call.assert_not_called()
        assert result["fix_suggestions"] == []

    @patch("agents.nodes.generate_fixes._call_groq")
    def test_invalid_json_fallback(self, mock_call):
        """When LLM returns invalid JSON, fallback templates are used."""
        mock_call.return_value = "Sorry, I can't generate JSON right now."
        state = {
            "prioritized_issues": [
                {"label": "Bug", "confidence": 0.8, "code": "x = None; x.y"},
            ]
        }
        result = generate_fixes_node(state)
        assert len(result["fix_suggestions"]) == 1
        s = result["fix_suggestions"][0]
        assert s["explanation"] == "Potential issue detected by static analysis."
        assert s["severity"] == "Medium"
