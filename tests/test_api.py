"""
Tests for the FastAPI backend endpoints.
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


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestTestDiffEndpoint:
    def test_test_diff_returns_review(self):
        response = client.post("/test-diff")
        assert response.status_code == 200
        data = response.json()
        assert "review" in data
        assert "issues_found" in data
        assert isinstance(data["issues_found"], int)


class TestReviewEndpoint:
    def test_review_with_valid_diff(self):
        diff_body = {
            "diff": (
                "--- a/test.py\n"
                "+++ b/test.py\n"
                "@@ -1,3 +1,3 @@\n"
                "-x = None\n"
                "-print(x.value)\n"
                "+if x is not None:\n"
                "+    print(x.value)\n"
            )
        }
        response = client.post("/review", json=diff_body)
        assert response.status_code == 200
        data = response.json()
        assert "review" in data

    def test_review_with_empty_diff(self):
        response = client.post("/review", json={"diff": ""})
        assert response.status_code == 400
