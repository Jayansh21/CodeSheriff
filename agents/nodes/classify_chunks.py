"""
Agent Node — classify_chunks

Runs the fine-tuned CodeBERT classifier on each code chunk.
Falls back to heuristic classification if the model is unavailable.
"""

import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger
from utils.config import LABEL_NAMES

logger = get_logger("agents.nodes.classify_chunks")


# If the model's confidence is below this threshold, prefer the heuristic.
_CONFIDENCE_THRESHOLD = 0.50


def _heuristic_classify(code: str) -> dict:
    """
    Fallback heuristic classifier used when the trained model is not
    available or its confidence is too low.  Mirrors the extended
    labelling logic in ml/dataset.py.
    """
    import re

    # Security Vulnerability
    sec_patterns = [
        r"""['"]SELECT\s.*?\+\s""",
        r"""f['"]\s*SELECT""",
        r"\.format\(.*?SELECT",
        r"\bcursor\.execute\s*\([^)]*\+",
        r"\beval\s*\(",
        r"\bos\.system\s*\([^)]*\+",
        r"\bsubprocess\.\w+\([^)]*shell\s*=\s*True",
    ]
    if any(re.search(p, code, re.IGNORECASE) for p in sec_patterns):
        return {"label": LABEL_NAMES[3], "confidence": 0.85, "label_id": 3}

    # Null Reference Risk
    null_patterns = [
        r"=\s*None[\s\S]{0,80}\.\w+",
        r"\.fetchone\(\)\.\w+",
        r"return\s+\w+\[.*?\]\.\w+",
        r"\.get\s*\([^)]+\)\.\w+",
    ]
    if any(re.search(p, code) for p in null_patterns):
        return {"label": LABEL_NAMES[1], "confidence": 0.82, "label_id": 1}

    # Type Mismatch
    type_patterns = [
        r"\bif\s+\w+\s*=[^=]",
        r'"\s*\+\s*\w+(?!\s*\()',
    ]
    if any(re.search(p, code) for p in type_patterns):
        return {"label": LABEL_NAMES[2], "confidence": 0.80, "label_id": 2}

    # Logic Flaw
    logic_patterns = [
        r"range\(len\(\w+\)\s*\+\s*1\)",
        r"/\s*0\b",
        r"while\s+True\b(?![\s\S]{0,200}break)",
        r"\[\s*len\(\w+\)\s*\]",
    ]
    if any(re.search(p, code) for p in logic_patterns):
        return {"label": LABEL_NAMES[4], "confidence": 0.78, "label_id": 4}

    # Clean
    return {"label": LABEL_NAMES[0], "confidence": 0.70, "label_id": 0}


def classify_chunks_node(state: dict) -> dict:
    """LangGraph node: classify each code chunk."""
    chunks: List[str] = state.get("code_chunks", [])
    classifications = []

    # Try to use the trained model; fall back to heuristic
    try:
        from ml.inference import predict_bug
        use_model = True
        logger.info("Using fine-tuned model for classification.")
    except Exception:
        use_model = False
        logger.warning("Fine-tuned model unavailable — using heuristic classifier.")

    for i, chunk in enumerate(chunks):
        if use_model:
            try:
                result = predict_bug(chunk)
                # If the model is uncertain, prefer the heuristic
                if result.get("confidence", 0) < _CONFIDENCE_THRESHOLD:
                    heuristic = _heuristic_classify(chunk)
                    if heuristic["label_id"] != 0:  # heuristic found something
                        logger.debug(
                            f"Chunk {i}: model low-confidence "
                            f"({result['confidence']:.2f}), using heuristic."
                        )
                        result = heuristic
            except Exception as e:
                logger.warning(f"Model inference failed for chunk {i}: {e}. Falling back.")
                result = _heuristic_classify(chunk)
        else:
            result = _heuristic_classify(chunk)

        result["chunk_index"] = i
        result["code"] = chunk
        classifications.append(result)
        logger.debug(f"Chunk {i}: {result['label']} ({result['confidence']:.2f})")

    logger.info(f"Classified {len(classifications)} chunk(s).")
    return {"classifications": classifications}
