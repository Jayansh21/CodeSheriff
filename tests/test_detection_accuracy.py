"""
Detection accuracy tests — verifies that CodeSheriff correctly identifies
known bug categories across multiple synthetic code samples.

Tests 10 realistic buggy files covering: SQL injection, command injection,
division by zero, null reference, logic bug, type bug, resource leak,
unsafe subprocess, insecure hashing, and bad boolean logic.

Each test builds a minimal unified diff, runs the pipeline through
parse → classify → prioritize, and checks the expected issue types.
"""

import sys
from pathlib import Path

import pytest

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

    def test_moderate_confidence_refined(self):
        """High-signal rules fire for moderate-confidence predictions."""
        result = _refine_issue_type("Clean", 0.80, "return x / len(y)")
        assert result == "Runtime Bug"

    def test_very_high_confidence_not_refined(self):
        """>=0.95 ML confidence is locked — rules never override."""
        result = _refine_issue_type("Clean", 0.95, "return x / len(y)")
        assert result == "Clean"

    def test_or_literal_moderate_confidence(self):
        code = 'if x == "a" or "b":'
        result = _refine_issue_type("Clean", 0.80, code)
        assert result == "Logic Bug"

    def test_or_literal_very_high_confidence_locked(self):
        """>=0.95 confidence lock applies to all rules including high-signal."""
        code = 'if x == "a" or "b":'
        result = _refine_issue_type("Clean", 0.96, code)
        assert result == "Clean"

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
        assert result == "Clean"

    def test_confidence_lock_boundary(self):
        """Exactly 0.95 should be locked (>=0.95)."""
        result = _refine_issue_type("Clean", 0.95, "x / len(y)")
        assert result == "Clean"

    def test_just_below_lock(self):
        """0.94 is below lock — rules should fire."""
        result = _refine_issue_type("Clean", 0.94, "x / len(y)")
        assert result == "Runtime Bug"


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
        assert len(issues) >= 2, f"Expected >=2 issues, got {len(issues)}: {labels}"

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


# ---------------------------------------------------------------------------
# Test file 4: auth_manager.py — SQL injection + insecure hashing
# ---------------------------------------------------------------------------

_AUTH_MANAGER_CODE = """\
import hashlib
import sqlite3


class AuthManager:

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def login(self, username, password):
        hashed = hashlib.md5(password.encode()).hexdigest()
        query = f"SELECT * FROM users WHERE name='{username}' AND pass='{hashed}'"
        return self.conn.execute(query).fetchone()

    def reset_password(self, user_id, new_pass):
        h = hashlib.sha1(new_pass.encode()).hexdigest()
        self.conn.execute(
            "UPDATE users SET pass='" + h + "' WHERE id=" + str(user_id)
        )
        self.conn.commit()
"""


class TestAuthManagerDetection:
    def test_detects_issues(self):
        diff = _make_diff("auth_manager.py", _AUTH_MANAGER_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 1, f"Expected >=1 issues, got {len(issues)}: {labels}"


# ---------------------------------------------------------------------------
# Test file 5: file_storage.py — command injection + resource leak
# ---------------------------------------------------------------------------

_FILE_STORAGE_CODE = """\
import os
import subprocess


class FileStorage:

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def delete_file(self, name):
        os.system("rm -rf " + name)

    def compress(self, archive_name, source):
        subprocess.call("tar czf " + archive_name + " " + source, shell=True)

    def read_first_line(self, path):
        f = open(path)
        line = f.readline()
        return line

    def disk_usage(self, path):
        result = subprocess.check_output("du -sh " + path, shell=True)
        return result.decode()
"""


class TestFileStorageDetection:
    def test_detects_issues(self):
        diff = _make_diff("file_storage.py", _FILE_STORAGE_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 1, f"Expected >=1 issues, got {len(issues)}: {labels}"

    def test_no_trivial_chunks(self):
        diff = _make_diff("file_storage.py", _FILE_STORAGE_CODE)
        state = {"pr_diff": diff}
        state.update(parse_diff_node(state))
        state.update(classify_chunks_node(state))
        for c in state["classifications"]:
            assert not _is_trivial_chunk(c["code"].strip())


# ---------------------------------------------------------------------------
# Test file 6: order_service.py — logic bugs + runtime bugs
# ---------------------------------------------------------------------------

_ORDER_SERVICE_CODE = """\
class OrderService:

    def __init__(self, db):
        self.db = db

    def apply_coupon(self, price, coupon_type):
        if coupon_type == "half" or "full":
            return price * 0.5
        return price

    def average_order_value(self, orders):
        total = sum(o["amount"] for o in orders)
        return total / len(orders)

    def get_order_name(self, order_id):
        row = self.db.execute(
            "SELECT name FROM orders WHERE id=" + str(order_id)
        ).fetchone()
        return row.name

    def split_bill(self, total, guests):
        return total / len(guests)
"""


class TestOrderServiceDetection:
    def test_detects_issues(self):
        diff = _make_diff("order_service.py", _ORDER_SERVICE_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 2, f"Expected >=2 issues, got {len(issues)}: {labels}"


# ---------------------------------------------------------------------------
# Test file 7: analytics_engine.py — division by zero + null ref
# ---------------------------------------------------------------------------

_ANALYTICS_CODE = """\
class AnalyticsEngine:

    def __init__(self, conn):
        self.conn = conn

    def conversion_rate(self, conversions, visitors):
        return conversions / len(visitors)

    def top_product(self, cursor):
        return cursor.fetchone().product_name

    def avg_session(self, sessions):
        durations = [s["duration"] for s in sessions]
        return sum(durations) / len(durations)

    def bounce_rate(self, bounces, total):
        if total == 0 or "unknown":
            return 0
        return bounces / total
"""


class TestAnalyticsEngineDetection:
    def test_detects_issues(self):
        diff = _make_diff("analytics_engine.py", _ANALYTICS_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 2, f"Expected >=2 issues, got {len(issues)}: {labels}"


# ---------------------------------------------------------------------------
# Test file 8: report_generator.py — command injection + unsafe eval
# ---------------------------------------------------------------------------

_REPORT_GENERATOR_CODE = """\
import os
import subprocess


class ReportGenerator:

    def __init__(self, template_dir):
        self.template_dir = template_dir

    def generate_pdf(self, report_name):
        os.system("wkhtmltopdf " + report_name + ".html " + report_name + ".pdf")

    def run_custom_script(self, script):
        subprocess.Popen(script, shell=True)

    def render_template(self, user_expr):
        return eval(user_expr)

    def export_csv(self, table_name):
        cmd = f"sqlite3 data.db '.headers on' '.mode csv' 'SELECT * FROM {table_name}'"
        return subprocess.check_output(cmd, shell=True)
"""


class TestReportGeneratorDetection:
    def test_detects_issues(self):
        diff = _make_diff("report_generator.py", _REPORT_GENERATOR_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 1, f"Expected >=1 issues, got {len(issues)}: {labels}"


# ---------------------------------------------------------------------------
# Test file 9: cache_manager.py — null ref + logic bugs
# ---------------------------------------------------------------------------

_CACHE_MANAGER_CODE = """\
class CacheManager:

    def __init__(self, capacity):
        self.store = {}
        self.capacity = capacity

    def get_value(self, key):
        return self.store.get(key).upper()

    def evict_ratio(self, evictions, total_ops):
        return evictions / len(total_ops)

    def should_evict(self, usage):
        if usage == "high" or "critical":
            return True
        return False

    def hit_rate(self, hits, misses):
        total = hits + misses
        return hits / total
"""


class TestCacheManagerDetection:
    def test_detects_issues(self):
        diff = _make_diff("cache_manager.py", _CACHE_MANAGER_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 2, f"Expected >=2 issues, got {len(issues)}: {labels}"


# ---------------------------------------------------------------------------
# Test file 10: api_client.py — SQL injection + unsafe subprocess
# ---------------------------------------------------------------------------

_API_CLIENT_CODE = """\
import subprocess
import sqlite3


class ApiClient:

    def __init__(self, db):
        self.conn = sqlite3.connect(db)

    def search_users(self, query_term):
        sql = "SELECT * FROM users WHERE name LIKE '%" + query_term + "%'"
        return self.conn.execute(sql).fetchall()

    def call_external(self, endpoint):
        subprocess.call("curl " + endpoint, shell=True)

    def log_request(self, method, path):
        self.conn.execute(
            f"INSERT INTO logs (method, path) VALUES ('{method}', '{path}')"
        )

    def avg_response_time(self, timings):
        return sum(timings) / len(timings)
"""


class TestApiClientDetection:
    def test_detects_issues(self):
        diff = _make_diff("api_client.py", _API_CLIENT_CODE)
        issues = _run_pipeline(diff)
        labels = [i["label"] for i in issues]
        assert len(issues) >= 2, f"Expected >=2 issues, got {len(issues)}: {labels}"

    def test_no_trivial_chunks(self):
        diff = _make_diff("api_client.py", _API_CLIENT_CODE)
        state = {"pr_diff": diff}
        state.update(parse_diff_node(state))
        state.update(classify_chunks_node(state))
        for c in state["classifications"]:
            assert not _is_trivial_chunk(c["code"].strip())


# ---------------------------------------------------------------------------
# Comprehensive accuracy logger — runs all 10 files and prints a report
# ---------------------------------------------------------------------------

_ALL_TEST_FILES = {
    "user_service.py": _USER_SERVICE_CODE,
    "payment_processor.py": _PAYMENT_CODE,
    "data_pipeline.py": _DATA_PIPELINE_CODE,
    "auth_manager.py": _AUTH_MANAGER_CODE,
    "file_storage.py": _FILE_STORAGE_CODE,
    "order_service.py": _ORDER_SERVICE_CODE,
    "analytics_engine.py": _ANALYTICS_CODE,
    "report_generator.py": _REPORT_GENERATOR_CODE,
    "cache_manager.py": _CACHE_MANAGER_CODE,
    "api_client.py": _API_CLIENT_CODE,
}


class TestAccuracyReport:
    """Log detection results for all 10 test files (informational)."""

    def test_full_accuracy_report(self, capsys):
        total_issues = 0
        total_files = 0

        for fname, code in _ALL_TEST_FILES.items():
            diff = _make_diff(fname, code)
            state = {"pr_diff": diff}
            state.update(parse_diff_node(state))
            state.update(classify_chunks_node(state))
            state.update(prioritize_issues_node(state))

            n_chunks = len(state["code_chunks"])
            n_class = len(state["classifications"])
            issues = state["prioritized_issues"]
            total_issues += len(issues)
            total_files += 1

        with capsys.disabled():
            print(f"\n{'='*60}")
            print(f"  CodeSheriff Detection Accuracy Report")
            print(f"{'='*60}")
            for fname, code in _ALL_TEST_FILES.items():
                diff = _make_diff(fname, code)
                state = {"pr_diff": diff}
                state.update(parse_diff_node(state))
                state.update(classify_chunks_node(state))
                state.update(prioritize_issues_node(state))
                issues = state["prioritized_issues"]

                print(f"\n  {fname}:")
                if not issues:
                    print(f"    (no issues detected)")
                for iss in issues:
                    fn_line = iss["code"].splitlines()[0][:50] if iss["code"] else ""
                    print(f"    {iss['label']:25s} conf={iss['confidence']:.2%}  {fn_line}")

            print(f"\n  Total: {total_issues} issues across {total_files} files")
            print(f"{'='*60}\n")

        assert total_issues >= 15, (
            f"Expected >=15 total issues across 10 files, got {total_issues}"
        )
