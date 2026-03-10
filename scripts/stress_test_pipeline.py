"""
CodeSheriff — Pipeline Stress Test

Runs the full LangGraph pipeline on multiple synthetic diffs to verify
stability and measure throughput. Skips the Groq LLM step to avoid
API costs — tests parse → classify → prioritize → format.

Usage (from project root):
    python scripts/stress_test_pipeline.py
"""

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.nodes.parse_diff import parse_diff_node
from agents.nodes.classify_chunks import classify_chunks_node
from agents.nodes.prioritize_issues import prioritize_issues_node
from agents.nodes.format_review import format_review_node
from utils.logger import get_logger

logger = get_logger("scripts.stress_test")

# ---------------------------------------------------------------------------
# Synthetic diffs
# ---------------------------------------------------------------------------
SYNTHETIC_DIFFS = [
    # 1. SQL injection
    (
        "--- a/db.py\n+++ b/db.py\n@@ -1,4 +1,4 @@\n"
        " import sqlite3\n"
        " def get_user(uid):\n"
        '-    query = "SELECT * FROM users WHERE id = " + uid\n'
        "+    query = \"SELECT * FROM users WHERE id = ?\"\n"
        "     conn = sqlite3.connect('db.sqlite')\n"
    ),
    # 2. Null reference
    (
        "--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,4 @@\n"
        " def load(path):\n"
        "-    config = None\n"
        "-    print(config.settings)\n"
        "+    config = load_config(path)\n"
        "+    if config:\n"
        "+        print(config.settings)\n"
    ),
    # 3. Off-by-one
    (
        "--- a/util.py\n+++ b/util.py\n@@ -1,4 +1,4 @@\n"
        " def total(items):\n"
        "     s = 0\n"
        "-    for i in range(len(items) + 1):\n"
        "+    for i in range(len(items)):\n"
        "         s += items[i]\n"
    ),
    # 4. Type mismatch
    (
        "--- a/check.py\n+++ b/check.py\n@@ -1,3 +1,3 @@\n"
        " def check(val):\n"
        "-    if val = 10:\n"
        "+    if val == 10:\n"
        "         return True\n"
    ),
    # 5. Command injection
    (
        "--- a/run.py\n+++ b/run.py\n@@ -1,3 +1,4 @@\n"
        " import os\n"
        " def run_cmd(cmd):\n"
        '-    os.system("ls " + cmd)\n'
        "+    import subprocess\n"
        "+    subprocess.run(['ls', cmd])\n"
    ),
    # 6. Clean → clean (no removed lines with bugs)
    (
        "--- a/math.py\n+++ b/math.py\n@@ -1,3 +1,3 @@\n"
        " def add(a, b):\n"
        "-    return a+b\n"
        "+    return a + b\n"
    ),
    # 7. Multiple issues in one diff
    (
        "--- a/service.py\n+++ b/service.py\n@@ -1,8 +1,8 @@\n"
        " import sqlite3\n"
        " def process(uid, items):\n"
        '-    q = "SELECT * FROM t WHERE id=" + uid\n'
        "+    q = \"SELECT * FROM t WHERE id=?\"\n"
        "-    for i in range(len(items) + 1):\n"
        "+    for i in range(len(items)):\n"
        "         print(items[i])\n"
    ),
    # 8. eval()
    (
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,3 +1,3 @@\n"
        " def evaluate(expr):\n"
        "-    return eval(expr)\n"
        "+    import ast\n"
        "+    return ast.literal_eval(expr)\n"
    ),
    # 9. Division by zero
    (
        "--- a/div.py\n+++ b/div.py\n@@ -1,3 +1,4 @@\n"
        " def ratio(a, b):\n"
        "-    return a / 0\n"
        "+    if b == 0:\n"
        "+        raise ValueError('b is 0')\n"
        "+    return a / b\n"
    ),
    # 10. fetchone null risk
    (
        "--- a/query.py\n+++ b/query.py\n@@ -1,4 +1,5 @@\n"
        " def get_name(conn):\n"
        "-    row = conn.execute('SELECT name FROM users').fetchone()\n"
        "-    return row.name\n"
        "+    row = conn.execute('SELECT name FROM users').fetchone()\n"
        "+    if row:\n"
        "+        return row.name\n"
    ),
    # 11. Large diff (repeated lines)
    (
        "--- a/big.py\n+++ b/big.py\n@@ -1,20 +1,20 @@\n"
        + "".join(
            f' def fn{i}():\n-    return eval("2+{i}")\n+    return 2+{i}\n'
            for i in range(10)
        )
    ),
    # 12. Empty diff
    "",
]


def run_pipeline_no_llm(diff: str) -> dict:
    """Run parse → classify → prioritize → format (skip LLM)."""
    state = {"pr_diff": diff}
    state.update(parse_diff_node(state))
    state.update(classify_chunks_node(state))
    state.update(prioritize_issues_node(state))
    # Mock fix suggestions (skip Groq API)
    state["fix_suggestions"] = [
        {**issue, "fix_suggestion": "[stress-test mock fix]"}
        for issue in state["prioritized_issues"]
    ]
    state.update(format_review_node(state))
    return state


def main():
    print("=" * 60)
    print("CodeSheriff — Pipeline Stress Test")
    print("=" * 60)

    total_diffs = len(SYNTHETIC_DIFFS)
    passed = 0
    failed = 0
    times = []

    for i, diff in enumerate(SYNTHETIC_DIFFS, 1):
        label = f"Diff {i:>2}/{total_diffs}"
        try:
            t0 = time.perf_counter()
            state = run_pipeline_no_llm(diff)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            review = state.get("final_review", "")
            issues = state.get("prioritized_issues", [])
            chunks = state.get("code_chunks", [])

            # Verify output structure
            assert isinstance(review, str), "review is not a string"
            assert len(review) > 0, "review is empty"
            assert "# CodeSheriff Review" in review, "missing review header"

            passed += 1
            print(f"  ✅ {label} | chunks={len(chunks):>2} | "
                  f"issues={len(issues):>2} | time={elapsed:.3f}s")

        except Exception as e:
            failed += 1
            print(f"  ❌ {label} | ERROR: {e}")

    # Summary
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0

    print()
    print("=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"  Total diffs:    {total_diffs}")
    print(f"  Passed:         {passed}")
    print(f"  Failed:         {failed}")
    print(f"  Avg time/diff:  {avg_time:.3f}s")
    print(f"  Min time:       {min_time:.3f}s")
    print(f"  Max time:       {max_time:.3f}s")
    print(f"  Total time:     {sum(times):.3f}s")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
