"""
Tests for the LangGraph agent pipeline.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import SAMPLE_DIFF
from agents.nodes.parse_diff import parse_diff_node
from agents.nodes.classify_chunks import classify_chunks_node
from agents.nodes.prioritize_issues import prioritize_issues_node
from agents.nodes.format_review import format_review_node


class TestParseDiff:
    """Tests for the diff parser node."""

    def test_extracts_chunks(self):
        state = {"pr_diff": SAMPLE_DIFF}
        result = parse_diff_node(state)
        assert "code_chunks" in result
        assert len(result["code_chunks"]) > 0

    def test_empty_diff(self):
        state = {"pr_diff": ""}
        result = parse_diff_node(state)
        assert result["code_chunks"] == []

    def test_no_diff_key(self):
        result = parse_diff_node({})
        assert result["code_chunks"] == []


class TestClassifyChunks:
    """Tests for the classification node (uses heuristic fallback)."""

    def test_classifies_known_bug(self):
        state = {
            "code_chunks": [
                'query = "SELECT * FROM users WHERE id = " + user_id',
            ]
        }
        result = classify_chunks_node(state)
        assert len(result["classifications"]) == 1
        assert result["classifications"][0]["label_id"] != 0  # not Clean

    def test_empty_chunks(self):
        state = {"code_chunks": []}
        result = classify_chunks_node(state)
        assert result["classifications"] == []


class TestPrioritizeIssues:
    """Tests for the prioritization node."""

    def test_filters_clean(self):
        state = {
            "classifications": [
                {"label_id": 0, "label": "Clean", "confidence": 0.9, "code": "x=1"},
                {"label_id": 3, "label": "Security Vulnerability", "confidence": 0.85, "code": "y=2"},
            ]
        }
        result = prioritize_issues_node(state)
        assert len(result["prioritized_issues"]) == 1
        assert result["prioritized_issues"][0]["label_id"] == 3

    def test_sorts_by_severity(self):
        state = {
            "classifications": [
                {"label_id": 4, "label": "Logic Flaw", "confidence": 0.8, "code": "a"},
                {"label_id": 3, "label": "Security Vulnerability", "confidence": 0.9, "code": "b"},
                {"label_id": 1, "label": "Null Reference Risk", "confidence": 0.7, "code": "c"},
            ]
        }
        result = prioritize_issues_node(state)
        ids = [i["label_id"] for i in result["prioritized_issues"]]
        assert ids[0] == 3  # Security first


class TestFormatReview:
    """Tests for the review formatter."""

    def test_no_issues(self):
        state = {"fix_suggestions": []}
        result = format_review_node(state)
        assert "No issues detected" in result["final_review"]

    def test_with_issues(self):
        state = {
            "fix_suggestions": [
                {
                    "label": "SQL Injection",
                    "confidence": 0.9,
                    "code": "bad code",
                    "fix_suggestion": "Use parameterized queries.",
                }
            ]
        }
        result = format_review_node(state)
        assert "## Issue 1" in result["final_review"]
        assert "SQL Injection" in result["final_review"]


class TestFullPipeline:
    """End-to-end test of the full agent pipeline (without Groq API)."""

    def test_pipeline_runs(self):
        """Run parse → classify → prioritize → format (skip LLM)."""
        state = {"pr_diff": SAMPLE_DIFF}

        state.update(parse_diff_node(state))
        state.update(classify_chunks_node(state))
        state.update(prioritize_issues_node(state))

        # Skip generate_fixes (requires API key) — mock it
        state["fix_suggestions"] = [
            {**issue, "fix_suggestion": "Mock fix."}
            for issue in state["prioritized_issues"]
        ]

        state.update(format_review_node(state))

        assert "final_review" in state
        assert len(state["final_review"]) > 0
        # Should find at least one issue in the sample diff
        assert state["final_review"].count("## Issue") >= 1
