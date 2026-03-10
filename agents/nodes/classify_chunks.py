"""
Agent Node — classify_chunks

Runs the fine-tuned CodeBERT classifier on each code chunk and emits
**multi-label** results: every label whose probability >= _MULTI_LABEL_THRESHOLD
is returned per chunk.

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

# If the model's top-label confidence is below this, prefer the heuristic.
_CONFIDENCE_THRESHOLD = 0.50

# Any label with probability >= this is included in the multi-label output.
_MULTI_LABEL_THRESHOLD = 0.40


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
    """LangGraph node: classify each code chunk (multi-label)."""
    raw_chunks = state.get("code_chunks", [])
    classifications = []

    # Try to use the trained model; fall back to heuristic
    try:
        from ml.inference import predict_bug
        use_model = True
        logger.info("Using fine-tuned model for classification.")
    except Exception:
        use_model = False
        logger.warning("Fine-tuned model unavailable — using heuristic classifier.")

    for i, chunk in enumerate(raw_chunks):
        # Support both old format (str) and new format (dict with 'code')
        if isinstance(chunk, dict):
            code = chunk.get("code", "")
            file_path = chunk.get("file", "unknown")
            start_line = chunk.get("start_line", 1)
        else:
            code = chunk
            file_path = "unknown"
            start_line = 1

        # Skip overly large chunks (performance safeguard)
        if code.count("\n") > 100:
            logger.debug("Chunk %d skipped (>100 lines).", i)
            continue

        labels: List[dict] = []

        if use_model:
            try:
                result = predict_bug(code)
                top_conf = result.get("confidence", 0)

                # If model is uncertain, prefer heuristic
                if top_conf < _CONFIDENCE_THRESHOLD:
                    heuristic = _heuristic_classify(code)
                    if heuristic["label_id"] != 0:
                        logger.debug("Chunk %d: model low-confidence (%.2f), using heuristic.", i, top_conf)
                        labels = [{"type": heuristic["label"], "confidence": heuristic["confidence"], "label_id": heuristic["label_id"]}]
                    else:
                        labels = [{"type": result["label"], "confidence": top_conf, "label_id": result["label_id"]}]
                else:
                    # Multi-label: include all labels above threshold
                    all_probs = result.get("all_probs", None)
                    if all_probs:
                        for lid, prob in all_probs.items():
                            lid_int = int(lid)
                            if lid_int != 0 and prob >= _MULTI_LABEL_THRESHOLD:
                                labels.append({"type": LABEL_NAMES.get(lid_int, f"Unknown({lid_int})"), "confidence": round(prob, 4), "label_id": lid_int})
                    # Fallback: if no all_probs or none above threshold, use top label
                    if not labels:
                        labels = [{"type": result["label"], "confidence": top_conf, "label_id": result["label_id"]}]
            except Exception as e:
                logger.warning("Model inference failed for chunk %d: %s. Falling back.", i, e)
                heuristic = _heuristic_classify(code)
                labels = [{"type": heuristic["label"], "confidence": heuristic["confidence"], "label_id": heuristic["label_id"]}]
        else:
            heuristic = _heuristic_classify(code)
            labels = [{"type": heuristic["label"], "confidence": heuristic["confidence"], "label_id": heuristic["label_id"]}]

        for lbl in labels:
            classifications.append({
                "label": lbl["type"],
                "confidence": lbl["confidence"],
                "label_id": lbl["label_id"],
                "chunk_index": i,
                "code": code,
                "file": file_path,
                "start_line": start_line,
            })

        logger.debug("Chunk %d: %s", i, ", ".join(f"{l['type']} ({l['confidence']:.2f})" for l in labels))

    logger.info("Classified %d chunk(s) → %d classification(s).", len(raw_chunks), len(classifications))
    return {"classifications": classifications}
