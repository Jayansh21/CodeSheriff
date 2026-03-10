"""
Tests for ml/inference.py — predict_bug function.

NOTE: These tests work regardless of whether the trained model exists.
If the model is not present, they verify graceful error handling.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import MODELS_DIR


class TestPredictBug:
    """Tests for the predict_bug inference function."""

    def test_empty_input_returns_clean(self):
        from ml.inference import predict_bug

        result = predict_bug("")
        assert result["label_id"] == 0
        assert result["confidence"] == 0.0
        assert result["label"] == "Clean"

    def test_whitespace_only_returns_clean(self):
        from ml.inference import predict_bug

        result = predict_bug("   \n\t  ")
        assert result["label_id"] == 0
        assert result["confidence"] == 0.0

    def test_return_format(self):
        """Ensure the return dict always has the required keys."""
        from ml.inference import predict_bug

        result = predict_bug("x = 1")
        assert "label" in result
        assert "confidence" in result
        assert "label_id" in result

    @pytest.mark.skipif(
        not (MODELS_DIR / "final").exists(),
        reason="Trained model not available — skipping model-dependent test.",
    )
    def test_model_prediction_shape(self):
        """When the model IS available, verify output is valid."""
        from ml.inference import predict_bug

        code = 'query = "SELECT * FROM users WHERE id = " + user_id'
        result = predict_bug(code)
        assert isinstance(result["label_id"], int)
        assert 0 <= result["label_id"] <= 4
        assert 0.0 <= result["confidence"] <= 1.0
