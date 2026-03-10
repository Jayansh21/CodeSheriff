"""
CodeSheriff — Pipeline Runner Script

Loads the sample diff, runs the full agent pipeline, and prints the review
along with a summary line.

Usage (from project root):
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import SAMPLE_DIFF
from utils.logger import get_logger
from agents.graph import run_review

logger = get_logger("scripts.run_pipeline")


def main():
    logger.info("=" * 60)
    logger.info("CodeSheriff — Full Pipeline Run")
    logger.info("=" * 60)

    review = run_review(SAMPLE_DIFF)
    print("\n" + review)

    # Count issues
    issue_count = review.count("## Issue")
    print("\n" + "=" * 60)
    print(f"CodeSheriff found {issue_count} issue(s).")
    print("=" * 60)


if __name__ == "__main__":
    main()
