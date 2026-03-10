"""
Tests for ml/dataset.py — heuristic labelling logic.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset import assign_label


class TestAssignLabel:
    """Verify the heuristic labeller maps known patterns to correct labels."""

    def test_sql_injection(self):
        code = 'query = "SELECT * FROM users WHERE id = " + user_id'
        assert assign_label(code) == 3  # Security Vulnerability

    def test_null_reference(self):
        code = "result = conn.execute(query)\nreturn result.fetchone().name"
        assert assign_label(code) == 1  # Null Reference Risk

    def test_type_mismatch(self):
        code = "if discount = 100:\n    return 0"
        assert assign_label(code) == 2  # Type Mismatch

    def test_logic_flaw(self):
        code = "for i in range(len(items) + 1):\n    total += items[i]"
        assert assign_label(code) == 4  # Logic Flaw

    def test_clean_code(self):
        code = "def add(a, b):\n    return a + b"
        assert assign_label(code) == 0  # Clean

    def test_empty_code(self):
        assert assign_label("") == 0  # Clean

    def test_security_trumps_null_ref(self):
        """Security should be prioritised over null reference when both match."""
        code = (
            'query = "SELECT * FROM users WHERE id = " + user_id\n'
            "result = conn.execute(query)\n"
            "return result.fetchone().name"
        )
        assert assign_label(code) == 3  # Security Vulnerability (higher priority)
