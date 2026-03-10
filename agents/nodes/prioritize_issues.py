"""
Agent Node — prioritize_issues

Filters out clean classifications and sorts issues by severity.
"""

import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("agents.nodes.prioritize_issues")

# Severity ordering: higher = more critical
# Decision: Security bugs first (can be exploited), then null refs (crash risk),
# then type mismatches, then logic flaws.
_SEVERITY_ORDER = {
    3: 4,   # Security Vulnerability
    1: 3,   # Null Reference Risk
    2: 2,   # Type Mismatch
    4: 1,   # Logic Flaw
    0: 0,   # Clean
}


def prioritize_issues_node(state: dict) -> dict:
    """LangGraph node: remove clean results and sort by severity."""
    classifications: List[dict] = state.get("classifications", [])

    # Filter out clean (label_id == 0)
    issues = [c for c in classifications if c.get("label_id", 0) != 0]

    # Sort by severity (descending), then by confidence (descending)
    issues.sort(
        key=lambda x: (
            _SEVERITY_ORDER.get(x.get("label_id", 0), 0),
            x.get("confidence", 0),
        ),
        reverse=True,
    )

    logger.info(
        "Prioritised %d issue(s) from %d classification(s).",
        len(issues), len(classifications),
    )
    return {"prioritized_issues": issues}
