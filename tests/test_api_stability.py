"""
CodeSheriff — API Stability Tests

Sends multiple requests to /review via FastAPI TestClient to verify:
- consistent 200 status codes
- correct JSON schema
- no crashes under repeated load

Run with:
    python -m pytest tests/test_api_stability.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# 20 diverse diff payloads
# ---------------------------------------------------------------------------
DIFF_PAYLOADS = [
    # 1 — SQL injection
    "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n"
    '-q = "SELECT * FROM t WHERE id=" + uid\n'
    "+q = \"SELECT * FROM t WHERE id=?\"\n",

    # 2 — null ref
    "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,4 @@\n"
    " def f():\n-    x = None\n-    x.run()\n"
    "+    x = init()\n+    if x:\n+        x.run()\n",

    # 3 — off-by-one
    "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n"
    " def f(items):\n-    for i in range(len(items)+1):\n"
    "+    for i in range(len(items)):\n         print(items[i])\n",

    # 4 — type mismatch
    "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n"
    " def f(x):\n-    if x = 5:\n+    if x == 5:\n",

    # 5 — eval
    "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n"
    " def f(e):\n-    return eval(e)\n+    return safe_eval(e)\n",

    # 6 — clean
    "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n"
    " def add(a,b):\n-    return a+b\n+    return a + b\n",

    # 7 — os.system
    "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,3 @@\n"
    " import os\n-os.system('rm ' + path)\n"
    "+import subprocess\n+subprocess.run(['rm', path])\n",

    # 8 — fetchone
    "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,3 @@\n"
    " def f(c):\n-    return c.execute(q).fetchone().name\n"
    "+    row = c.execute(q).fetchone()\n+    return row.name if row else None\n",

    # 9 — f-string sql
    "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n"
    " def f(t):\n-    sql = f\"SELECT * FROM {t}\"\n+    sql = \"SELECT * FROM ?\"\n",

    # 10 — division
    "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,3 @@\n"
    " def f(a):\n-    return a / 0\n+    if a:\n+        return a\n",

    # 11-20: duplicates with minor variations to test stability
    "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
    '-q = "INSERT INTO t VALUES(" + v + ")"\n+q = "INSERT INTO t VALUES(?)"\n',

    "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
    " def f():\n-    return items[len(items)]\n+    return items[-1]\n",

    "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,3 @@\n"
    " def f(d):\n-    d.get('k').strip()\n+    v = d.get('k')\n+    if v: v.strip()\n",

    "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
    " def f(n):\n-    msg = 'count: ' + n\n+    msg = f'count: {n}'\n",

    "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n"
    " import subprocess\n def f(c):\n-    subprocess.run(c, shell=True)\n"
    "+    subprocess.run(c.split())\n",

    "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
    " def f():\n-    open(path).read()\n+    with open(path) as fp: fp.read()\n",

    "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
    " def f(a,b):\n-    return a+b\n+    return a + b\n",

    "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n"
    " def f(lst):\n     t=0\n-    for i in range(len(lst)+1): t+=lst[i]\n"
    "+    for i in range(len(lst)): t+=lst[i]\n",

    "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,3 @@\n"
    " def f():\n-    cfg=None\n-    cfg.x()\n+    cfg=load()\n+    if cfg: cfg.x()\n",

    "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
    '-q = "DELETE FROM t WHERE id=" + i\n+q = "DELETE FROM t WHERE id=?"\n',
]


class TestAPIStability:
    """Send 20 requests to /review and verify consistent behavior."""

    @pytest.mark.parametrize("diff", DIFF_PAYLOADS,
                             ids=[f"diff-{i+1}" for i in range(len(DIFF_PAYLOADS))])
    def test_review_returns_200(self, diff):
        response = client.post("/review", json={"diff": diff})
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:200]}"
        )

    @pytest.mark.parametrize("diff", DIFF_PAYLOADS,
                             ids=[f"schema-{i+1}" for i in range(len(DIFF_PAYLOADS))])
    def test_review_json_schema(self, diff):
        response = client.post("/review", json={"diff": diff})
        data = response.json()
        assert "review" in data, "Missing 'review' key"
        assert "issues_found" in data, "Missing 'issues_found' key"
        assert isinstance(data["review"], str), "review is not a string"
        assert isinstance(data["issues_found"], int), "issues_found is not an int"
        assert data["issues_found"] >= 0, "issues_found is negative"
        assert len(data["review"]) > 0, "review is empty"

    def test_empty_diff_returns_400(self):
        response = client.post("/review", json={"diff": ""})
        assert response.status_code == 400

    def test_missing_body_returns_422(self):
        response = client.post("/review")
        assert response.status_code == 422

    def test_health_during_load(self):
        """Health endpoint stays responsive between review requests."""
        for _ in range(5):
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
