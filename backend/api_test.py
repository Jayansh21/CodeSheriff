"""
CodeSheriff — API Integration Test

Hits the local FastAPI server and validates responses.

Usage (from project root, with server running):
    python backend/api_test.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print(f"✅ /health → {data}")


def test_test_diff():
    r = httpx.post(f"{BASE_URL}/test-diff", timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert "review" in data
    assert data["issues_found"] >= 0
    print(f"✅ /test-diff → {data['issues_found']} issue(s) found")
    print(data["review"][:500])


def test_review_endpoint():
    diff_body = {
        "diff": (
            '--- a/test.py\n'
            '+++ b/test.py\n'
            '@@ -1,3 +1,3 @@\n'
            '-x = None\n'
            '-print(x.value)\n'
            '+if x is not None:\n'
            '+    print(x.value)\n'
        )
    }
    r = httpx.post(f"{BASE_URL}/review", json=diff_body, timeout=60)
    assert r.status_code == 200
    data = r.json()
    print(f"✅ /review → {data['issues_found']} issue(s) found")


if __name__ == "__main__":
    print("=" * 50)
    print("CodeSheriff API Integration Tests")
    print("=" * 50)
    test_health()
    test_test_diff()
    test_review_endpoint()
    print("\nAll API tests passed ✅")
