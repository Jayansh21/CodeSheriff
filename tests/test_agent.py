"""
Tests for the LangGraph agent pipeline.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import SAMPLE_DIFF, CONFIDENCE_THRESHOLD
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


# ---------------------------------------------------------------------------
# Diff parsing: function-level chunking & line numbers
# ---------------------------------------------------------------------------

class TestDiffParsing:
    """Detailed diff parsing tests."""

    def test_function_level_splitting(self):
        """Each top-level def becomes its own chunk."""
        diff = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -0,0 +1,8 @@\n"
            "+def alpha():\n"
            "+    return 1\n"
            "+\n"
            "+def beta():\n"
            "+    return 2\n"
            "+\n"
            "+class Gamma:\n"
            "+    pass\n"
        )
        result = parse_diff_node({"pr_diff": diff})
        chunks = result["code_chunks"]
        assert len(chunks) == 3
        assert "alpha" in chunks[0]["code"]
        assert "beta" in chunks[1]["code"]
        assert "Gamma" in chunks[2]["code"]

    def test_start_line_tracking(self):
        """Chunks carry the correct start line from the hunk header."""
        diff = (
            "--- a/bar.py\n"
            "+++ b/bar.py\n"
            "@@ -0,0 +1,4 @@\n"
            "+def first():\n"
            "+    pass\n"
            "+def second():\n"
            "+    pass\n"
        )
        result = parse_diff_node({"pr_diff": diff})
        chunks = result["code_chunks"]
        assert chunks[0]["start_line"] == 1
        assert chunks[1]["start_line"] == 3

    def test_multi_file_diff(self):
        """Chunks from different files are separated correctly."""
        diff = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def a_func():\n"
            "+    pass\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def b_func():\n"
            "+    pass\n"
        )
        result = parse_diff_node({"pr_diff": diff})
        files = {c["file"] for c in result["code_chunks"]}
        assert files == {"a.py", "b.py"}

    def test_deleted_lines_excluded(self):
        """Only + lines appear in chunks; - lines are excluded."""
        diff = (
            "--- a/c.py\n"
            "+++ b/c.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old_code = 1\n"
            "+new_code = 2\n"
        )
        result = parse_diff_node({"pr_diff": diff})
        for chunk in result["code_chunks"]:
            assert "old_code" not in chunk["code"]
            assert "new_code" in chunk["code"]


# ---------------------------------------------------------------------------
# Confidence gate behaviour
# ---------------------------------------------------------------------------

class TestConfidenceGate:
    """Verify the confidence threshold logic in classify_chunks_node."""

    def test_low_confidence_downgraded(self):
        """Chunks with confidence < threshold get downgraded to Code Quality."""
        state = {
            "code_chunks": [
                {
                    "code": "def process(data):\n    result = data['key'] + 1\n    return result",
                    "file": "test.py",
                    "start_line": 1,
                }
            ]
        }
        result = classify_chunks_node(state)
        for c in result["classifications"]:
            if c["confidence"] < CONFIDENCE_THRESHOLD and c["label"] not in ("Clean", "Code Quality"):
                assert False, f"Low-confidence chunk not downgraded: {c['label']} {c['confidence']}"

    def test_trivial_chunks_skipped(self):
        """Trivial chunks produce no classifications."""
        state = {
            "code_chunks": [
                {
                    "code": "import os\nimport sys\nfrom pathlib import Path",
                    "file": "imports.py",
                    "start_line": 1,
                }
            ]
        }
        result = classify_chunks_node(state)
        assert result["classifications"] == []


# ---------------------------------------------------------------------------
# Format review: inline comments and summary
# ---------------------------------------------------------------------------

class TestFormatReviewDetails:
    """Additional format_review_node tests."""

    def test_inline_comments_generated(self):
        state = {
            "fix_suggestions": [
                {
                    "label": "Security Vulnerability",
                    "confidence": 0.95,
                    "code": "query = 'SELECT * FROM users WHERE id=' + uid",
                    "fix_suggestion": "Use parameterized queries.",
                    "file": "app.py",
                    "start_line": 10,
                }
            ]
        }
        result = format_review_node(state)
        assert len(result["inline_comments"]) == 1
        ic = result["inline_comments"][0]
        assert ic["file"] == "app.py"
        assert ic["line"] >= 10
        assert "CodeSheriff" in ic["body"]

    def test_summary_grouped_counts(self):
        state = {
            "fix_suggestions": [
                {"label": "Security Vulnerability", "confidence": 0.9,
                 "code": "a", "fix_suggestion": "fix a", "file": "x.py", "start_line": 1},
                {"label": "Security Vulnerability", "confidence": 0.8,
                 "code": "b", "fix_suggestion": "fix b", "file": "x.py", "start_line": 5},
                {"label": "Runtime Bug", "confidence": 0.7,
                 "code": "c", "fix_suggestion": "fix c", "file": "y.py", "start_line": 1},
            ]
        }
        result = format_review_node(state)
        summary = result["review_summary"]
        assert "3 issues" in summary
        assert "Security Vulnerabilit" in summary
        assert "Runtime Bug" in summary

    def test_issue_line_offset_sql(self):
        """Line offset finder should locate the SQL injection line."""
        from agents.nodes.format_review import _find_issue_line_offset
        code = "def get_user(uid):\n    q = 'SELECT * FROM users WHERE id=' + uid\n    return db.execute(q)"
        offset = _find_issue_line_offset(code)
        assert offset == 1  # the SQL line is on line index 1


# ---------------------------------------------------------------------------
# Function chunking: boundary detection edge cases
# ---------------------------------------------------------------------------

class TestFunctionChunking:
    """Edge cases for function-level chunk splitting."""

    def test_async_def_boundary(self):
        """async def is treated as a function boundary."""
        diff = (
            "--- a/svc.py\n"
            "+++ b/svc.py\n"
            "@@ -0,0 +1,6 @@\n"
            "+async def fetch():\n"
            "+    return await get()\n"
            "+\n"
            "+async def save(data):\n"
            "+    await put(data)\n"
            "+    return True\n"
        )
        result = parse_diff_node({"pr_diff": diff})
        chunks = result["code_chunks"]
        assert len(chunks) == 2
        assert "fetch" in chunks[0]["code"]
        assert "save" in chunks[1]["code"]

    def test_chunk_carries_file_path(self):
        """Every chunk records the originating file."""
        diff = (
            "--- a/my/module.py\n"
            "+++ b/my/module.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+def helper():\n"
            "+    return 42\n"
            "+\n"
        )
        result = parse_diff_node({"pr_diff": diff})
        assert result["code_chunks"][0]["file"] == "my/module.py"

    def test_large_chunk_capped(self):
        """Chunks exceeding _MAX_CHUNK_LINES are split."""
        from agents.nodes.parse_diff import _MAX_CHUNK_LINES
        body = "\n".join(f"+    x_{i} = {i}" for i in range(_MAX_CHUNK_LINES + 10))
        diff = (
            "--- a/big.py\n"
            "+++ b/big.py\n"
            f"@@ -0,0 +1,{_MAX_CHUNK_LINES + 11} @@\n"
            "+def big_func():\n"
            f"{body}\n"
        )
        result = parse_diff_node({"pr_diff": diff})
        for chunk in result["code_chunks"]:
            lines = chunk["code"].splitlines()
            assert len(lines) <= _MAX_CHUNK_LINES


# ---------------------------------------------------------------------------
# Issue-type resolution & label mapping
# ---------------------------------------------------------------------------

class TestIssueTypeMapping:
    """Unit tests for _resolve_issue_type."""

    def test_known_labels_mapped(self):
        from agents.nodes.classify_chunks import _resolve_issue_type
        assert _resolve_issue_type("Security Vulnerability") == "Security Vulnerability"
        assert _resolve_issue_type("Null Reference Risk") == "Runtime Bug"
        assert _resolve_issue_type("Type Mismatch") == "Type Bug"
        assert _resolve_issue_type("Logic Flaw") == "Logic Bug"
        assert _resolve_issue_type("Clean") == "Clean"

    def test_unknown_label_fallback(self):
        from agents.nodes.classify_chunks import _resolve_issue_type
        assert _resolve_issue_type("SomethingNew") == "Code Issue"


# ---------------------------------------------------------------------------
# Format review: fix-section parser
# ---------------------------------------------------------------------------

class TestFixSectionParser:
    """Unit tests for _parse_fix_sections in format_review."""

    def test_extracts_code_block(self):
        from agents.nodes.format_review import _parse_fix_sections
        text = "This is bad.\n\n**Suggested Fix**\n```python\nx = safe()\n```\n"
        sections = _parse_fix_sections(text)
        assert "x = safe()" in sections["suggested_fix"]

    def test_plain_text_preserved(self):
        from agents.nodes.format_review import _parse_fix_sections
        text = "Use parameterized queries to avoid SQL injection."
        sections = _parse_fix_sections(text)
        assert sections["explanation"] == text
