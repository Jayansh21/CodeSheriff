"""
CodeSheriff — LangGraph Agent Pipeline

Wires the five processing nodes into a sequential graph:

    parse_diff → classify_chunks → prioritize_issues →
    generate_fixes → format_review

Usage (from project root):
    python -m agents.graph
"""

import sys
from pathlib import Path
from typing import List, TypedDict

from langgraph.graph import StateGraph, END

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.nodes.parse_diff import parse_diff_node
from agents.nodes.classify_chunks import classify_chunks_node
from agents.nodes.prioritize_issues import prioritize_issues_node
from agents.nodes.generate_fixes import generate_fixes_node
from agents.nodes.format_review import format_review_node
from utils.logger import get_logger

logger = get_logger("agents.graph")


# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------
class ReviewState(TypedDict):
    pr_diff: str
    code_chunks: List[dict]
    language: str
    languages: dict
    classifications: List[dict]
    prioritized_issues: List[dict]
    fix_suggestions: List[dict]
    final_review: str
    review_summary: str
    inline_comments: List[dict]


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_review_graph() -> StateGraph:
    """Construct and compile the CodeSheriff review pipeline."""

    graph = StateGraph(ReviewState)

    graph.add_node("parse_diff", parse_diff_node)
    graph.add_node("classify_chunks", classify_chunks_node)
    graph.add_node("prioritize_issues", prioritize_issues_node)
    graph.add_node("generate_fixes", generate_fixes_node)
    graph.add_node("format_review", format_review_node)

    graph.set_entry_point("parse_diff")
    graph.add_edge("parse_diff", "classify_chunks")
    graph.add_edge("classify_chunks", "prioritize_issues")
    graph.add_edge("prioritize_issues", "generate_fixes")
    graph.add_edge("generate_fixes", "format_review")
    graph.add_edge("format_review", END)

    return graph.compile()


def run_review(pr_diff: str) -> dict:
    """
    Run the full review pipeline and return the result dict containing
    ``final_review``, ``review_summary``, and ``inline_comments``.
    """
    app = build_review_graph()
    initial_state: ReviewState = {
        "pr_diff": pr_diff,
        "code_chunks": [],
        "language": "",
        "languages": {},
        "classifications": [],
        "prioritized_issues": [],
        "fix_suggestions": [],
        "final_review": "",
        "review_summary": "",
        "inline_comments": [],
    }
    result = app.invoke(initial_state)
    return {
        "final_review": result.get("final_review", ""),
        "review_summary": result.get("review_summary", ""),
        "inline_comments": result.get("inline_comments", []),
        "language": result.get("language", ""),
        "languages": result.get("languages", {}),
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils.config import SAMPLE_DIFF

    logger.info("Running CodeSheriff review pipeline …")
    result = run_review(SAMPLE_DIFF)
    print("\n" + result["final_review"])
