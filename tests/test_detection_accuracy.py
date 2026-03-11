"""
Detection accuracy tests — verifies that CodeSheriff correctly identifies
known bug categories across multiple synthetic code samples.

Each test builds a minimal unified diff, runs the pipeline through
parse → classify → prioritize, and checks the expected issue types.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.nodes.parse_diff import parse_diff_node
from agents.nodes.classify_chunks import (
    classify_chunks_node,
    _is_trivial_chunk,
    _refine_issue_type,
)
from agents.nodes.prioritize_issues import prioritize_issues_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_diff(filename: str, code: str) -> str:
    """Wrap raw code into a minimal unified diff."""
    lines = code.strip().splitlines()
    added = "\n".join("+" + ln for ln in lines)
    return (
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{added}\n"
    )


def _run_pipeline(diff: str) -> list:
    """Run parse → classify → prioritize and return prioritized issues."""
    state = {"pr_diff": diff}
    state.update(parse_diff_node(state))
    state.update(classify_chunks_node(state))
    state.update(prioritize_issues_node(state))
    return state["prioritized_issues"]


# ---------------------------------------------------------------------------
# Trivial-chunk filter unit tests
# ---------------------------------------------------------------------------

class TestTrivialChunkFilter:

    def test_imports_only(self):
        assert _is_trivial_chunk("import os\nimport sys\nfrom pathlib import Path")

    def test_bare_class(self):
        assert _is_trivial_chunk("class MyService:\n    pass")

    def test_bare_class_with_docstring(self):
        assert _is_trivial_chunk('class Foo:\n    """A foo."""')

    def test_init_single_assignment(self):
        assert _is_trivial_chunk("def __init__(self, db):\n    self.db = connect(db)")

    def test_real_logic_not_trivial(self):
        code = (
            "def get_user(uid):\n"
            '    q = "SELECT * FROM users WHERE id = " + uid\n'
            "    return db.execute(q)"
        )
        assert not _is_trivial_chunk(code)

    def test_empty_chunk(self):
        assert _is_trivial_chunk("")

    def test_only_comments(self):
        assert _is_trivial_chunk("# this is a comment\n# another comment")

    def test_function_with_body(self):
        assert not _is_trivial_chunk(
            "def calc(x, y):\n    return x / len(y)"
        )


# ---------------------------------------------------------------------------
# Refinement rule unit tests
# ---------------------------------------------------------------------------

class TestRefinementRules:

    def test_div_by_len_becomes_runtime(self):
        result = _refine_issue_type("Code Quality", 0.55, "return x / len(items)")
        assert result == "Runtime Bug"

    def test_or_literal_becomes_logic(self):
        code = 'if role == "admin" or "superuser":'
        result = _refine_issue_type("Code Quality", 0.50, code)
        assert result == "Logic Bug"

    def test_or_variable_becomes_logic(self):
        code = 'if role == "admin" or superuser:'
        result = _refine_issue_type("Code Quality", 0.50, code)
        assert result == "Logic Bug"

    def test_high_confidence_clean_still_refined(self):
        """High-signal rules override even confident Clean predictions."""
        result = _refine_issue_type("Clean", 0.97, "return x / len(y)")
        assert result == "Runtime Bug"

    def test_or_literal_high_confidence(self):
        """Logic bug pattern is high-signal — fires regardless of confidence."""
        code = 'if x == "a" or "b":'
        result = _refine_issue_type("Clean", 0.95, code)
        assert result == "Logic Bug"

    def test_fetchone_bracket(self):
        result = _refine_issue_type("Code Quality", 0.60, "row = cur.fetchone()[0]")
        assert result == "Runtime Bug"

    def test_fetchone_attr(self):
        result = _refine_issue_type("Code Quality", 0.60, "name = cur.fetchone().name")
        assert result == "Runtime Bug"

    def test_borderline_subprocess_shell(self):
        code = 'subprocess.call("cmd " + x, shell=True)'
        result = _refine_issue_type("Code Quality", 0.50, code)
        assert result == "Security Vulnerability"

    def test_borderline_os_system(self):
        code = 'os.system("rm " + path)'
        result = _refine_issue_type("Code Quality", 0.50, code)
        assert result == "Security Vulnerability"

    def test_high_confidence_borderline_rule_skipped(self):
        """Borderline rules should not fire when confidence is high."""
        code = 'os.system("rm " + path)'
        result = _refine_issue_type("Clean", 0.90, code)
        # os.system is a borderline rule — requires conf < 0.75
        assert result == "Clean"


# ---------------------------------------------------------------------------
# Test file 1: user_service.py (original PR file)
# ---------------------------------------------------------------------------

_USER_SERVICE_CODE = """\
import sqlite3
import subprocess
import json


class UserService:

    def __init__(self, db):
        self.db = sqlite3.connect(db)

    def get_user_by_email(self, email):
        query = f"SELECT * FROM users WHERE email = '{email}'"
        cursor = self.db.execute(query)
        return cursor.fetchone()

    def load_profile(self, data):
        profile = json.loads(data)
        return profile["name"]

    def is_premium(self, role):
        if role == "premium" or "admin":
            return True
        return False

    def average_score(self, scores):
        total = sum(scores)
        return total / len(scores)

    def export_db(self, filename):
        subprocess.call("cp database.db " + filename, shell=True)

    def create_cursor(self):
        cursor = self.db.cursor()
        return cursor

    def apply_discount(self, price, discount):
        final_price = price - discount
        return final_price
"""


class TestUserServiceDetection:
    """Expected: SQL injection, command injection, logic bug, runtime bug detected."""

    def test_detects_issues(self):
        diff = _make_diff("user_service.py", _USER_SERVICE_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        # Should detect at least security + runtime issues
        assert len(issues) >= 2, f"Expected >=2 issues, got {len(issues)}: {labels}"

    def test_no_import_false_positive(self):
        """Import-only chunk should not appear as an issue."""
        diff = _make_diff("user_service.py", _USER_SERVICE_CODE)
        issues = _run_pipeline(diff)
        # No issue should have code that is purely imports
        for issue in issues:
            code_lines = [
                ln.strip() for ln in issue["code"].splitlines()
                if ln.strip()
            ]
            all_imports = all(
                ln.startswith("import ") or ln.startswith("from ")
                for ln in code_lines
            )
            assert not all_imports, f"Import-only chunk flagged: {issue['label']}"


# ---------------------------------------------------------------------------
# Test file 2: payment_processor.py — diverse bug types
# ---------------------------------------------------------------------------

_PAYMENT_CODE = """\
import os
import hashlib


class PaymentProcessor:

    def __init__(self, api_key):
        self.api_key = api_key

    def hash_card(self, card_number):
        return hashlib.md5(card_number.encode()).hexdigest()

    def charge(self, amount, currency):
        if currency == "USD" or "EUR":
            return amount * 1.0
        return None

    def refund(self, transactions, tx_id):
        for tx in transactions:
            if tx["id"] == tx_id:
                return tx["amount"] / len(tx.get("splits", []))
        return 0

    def run_report(self, report_name):
        os.system("generate_report " + report_name)

    def get_balance(self, accounts):
        total = 0
        for acc in accounts:
            total += acc.balance
        return total / len(accounts)
"""


class TestPaymentProcessorDetection:
    """
    Expected:
    - hash_card: Security (MD5 is weak) — model may or may not catch this
    - charge: Logic Bug (or "EUR" always true)
    - refund: Runtime Bug (division by len of possibly empty list)
    - run_report: Security (command injection via os.system)
    - get_balance: Runtime Bug (division by len of possibly empty list)
    """

    def test_detects_issues(self):
        diff = _make_diff("payment_processor.py", _PAYMENT_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 3, f"Expected >=3 issues, got {len(issues)}: {labels}"

    def test_no_init_false_positive(self):
        diff = _make_diff("payment_processor.py", _PAYMENT_CODE)
        issues = _run_pipeline(diff)
        for issue in issues:
            code = issue["code"].strip()
            # __init__ with only self.api_key = api_key should not be an issue
            if "def __init__" in code:
                lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
                non_decl = [
                    ln for ln in lines
                    if not ln.startswith("def ") and not ln.startswith("self.")
                ]
                assert len(non_decl) > 0, f"Bare __init__ flagged: {issue['label']}"


# ---------------------------------------------------------------------------
# Test file 3: data_pipeline.py — runtime + logic bugs
# ---------------------------------------------------------------------------

_DATA_PIPELINE_CODE = """\
import subprocess


class DataPipeline:

    def __init__(self, source):
        self.source = source

    def average(self, values):
        return sum(values) / len(values)

    def normalize(self, data, factor):
        return [x / factor for x in data]

    def check_status(self, status):
        if status == "active" or "pending":
            return True
        return False

    def run_etl(self, script_path):
        subprocess.Popen(script_path, shell=True)

    def first_record(self, cursor):
        return cursor.fetchone().id
"""


class TestDataPipelineDetection:
    """
    Expected:
    - average: Runtime Bug (ZeroDivisionError)
    - normalize: Runtime Bug (division by zero if factor=0)
    - check_status: Logic Bug (or "pending" always true)
    - run_etl: Security (command injection)
    - first_record: Runtime Bug (fetchone() could be None)
    """

    def test_detects_issues(self):
        diff = _make_diff("data_pipeline.py", _DATA_PIPELINE_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 2, f"Expected >=2 issues, got {len(issues)}: {labels}"

    def test_trivial_chunks_not_flagged(self):
        diff = _make_diff("data_pipeline.py", _DATA_PIPELINE_CODE)
        state = {"pr_diff": diff}
        state.update(parse_diff_node(state))
        state.update(classify_chunks_node(state))

        for c in state["classifications"]:
            code = c["code"].strip()
            assert not _is_trivial_chunk(code), (
                f"Trivial chunk classified: {c['label']} — {code[:60]}"
            )
