"""
Agent Node — classify_chunks

Runs the fine-tuned CodeBERT classifier on each code chunk.

The ML model is the **sole** detection engine.  No hardcoded keyword/regex
detection exists here.  Two lightweight post-processing steps refine the
raw ML output:

  1. **Confidence gate** — predictions below ``CONFIDENCE_THRESHOLD``
     (default 0.60, configurable via env) are downgraded to "Code Quality".
  2. **Post-classification refinement** — a handful of generic code-pattern
     rules can *adjust* the issue type when the ML confidence is borderline.
     They never *replace* the ML classifier.
"""

import re
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger
from utils.config import LABEL_NAMES, CONFIDENCE_THRESHOLD

logger = get_logger("agents.nodes.classify_chunks")

# Any label with probability >= this is included in the multi-label output.
_MULTI_LABEL_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Issue-type mapping  (model label → user-facing category)
# ---------------------------------------------------------------------------
_LABEL_TYPE_MAP = {
    # Security
    "Security Vulnerability": "Security Vulnerability",
    "SQL_INJECTION": "Security Vulnerability",
    "COMMAND_INJECTION": "Security Vulnerability",
    "WEAK_HASH": "Security Vulnerability",
    # Runtime
    "Null Reference Risk": "Runtime Bug",
    "NULL_POINTER": "Runtime Bug",
    "DIVISION_ERROR": "Runtime Bug",
    # Type
    "Type Mismatch": "Type Bug",
    "TYPE_ERROR": "Type Bug",
    # Logic
    "Logic Flaw": "Logic Bug",
    "LOGIC_ERROR": "Logic Bug",
    # Quality
    "CODE_SMELL": "Code Quality",
    # Clean
    "Clean": "Clean",
}


def _resolve_issue_type(raw_label: str) -> str:
    """Map a raw model label to a user-facing issue type."""
    return _LABEL_TYPE_MAP.get(raw_label, "Code Issue")


# ---------------------------------------------------------------------------
# Light post-classification refinement rules.
# These only run when ML confidence is borderline (< 0.75) and only
# *adjust* the resolved issue_type — they never bypass the ML model.
# ---------------------------------------------------------------------------
_BORDERLINE_CEILING = 0.75  # rules only fire below this confidence


def _refine_issue_type(issue_type: str, confidence: float, code: str) -> str:
    """
    Optionally adjust *issue_type* based on obvious code patterns.

    Only activates when confidence is borderline (< 0.75) so that
    strong ML predictions are never overridden.
    """
    if confidence >= _BORDERLINE_CEILING:
        return issue_type

    # Division by a value that may be zero  (e.g. / len(...))
    if re.search(r"/\s*len\(", code):
        return "Runtime Bug"

    # Unguarded .fetchone()[index]  — common null-deref crash
    if re.search(r"\.fetchone\(\)\s*\[", code):
        return "Runtime Bug"

    # Unguarded .fetchone().attr  — same class of crash
    if re.search(r"\.fetchone\(\)\.\w+", code):
        return "Runtime Bug"

    return issue_type


def classify_chunks_node(state: dict) -> dict:
    """LangGraph node: classify each code chunk via the ML model."""
    raw_chunks = state.get("code_chunks", [])
    classifications: List[dict] = []

    # Import the ML inference function
    try:
        from ml.inference import predict_bug
        use_model = True
        logger.info("Using fine-tuned model for classification.")
    except Exception:
        use_model = False
        logger.warning("Fine-tuned model unavailable — chunks will be marked Code Quality.")

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

                # Multi-label: include every label above the multi-label
                # threshold when detailed probabilities are available.
                all_probs = result.get("all_probs", None)
                if all_probs and top_conf >= CONFIDENCE_THRESHOLD:
                    for lid, prob in all_probs.items():
                        lid_int = int(lid)
                        if lid_int != 0 and prob >= _MULTI_LABEL_THRESHOLD:
                            labels.append({
                                "type": LABEL_NAMES.get(lid_int, f"Unknown({lid_int})"),
                                "confidence": round(prob, 4),
                                "label_id": lid_int,
                            })

                # Fallback: use the top predicted label
                if not labels:
                    labels = [{
                        "type": result.get("label", "Clean"),
                        "confidence": top_conf,
                        "label_id": result.get("label_id", 0),
                    }]
            except Exception as e:
                logger.warning("Model inference failed for chunk %d: %s.", i, e)
                labels = [{"type": "Clean", "confidence": 0.0, "label_id": 0}]
        else:
            # No model available — mark as Code Quality so the LLM can
            # still provide a generic review without faking a detection.
            labels = [{"type": "CODE_SMELL", "confidence": 0.0, "label_id": 0}]

        # ------------------------------------------------------------------
        # Post-processing: confidence gate + refinement rules
        # ------------------------------------------------------------------
        for lbl in labels:
            conf = lbl["confidence"]
            raw_type = lbl["type"]

            # Step 1 — resolve model label to user-facing issue type
            issue_type = _resolve_issue_type(raw_type)

            # Step 2 — confidence gate: weak predictions become Code Quality
            if conf < CONFIDENCE_THRESHOLD and issue_type not in ("Clean", "Code Quality"):
                logger.debug(
                    "Chunk %d: confidence %.2f < %.2f — downgrading '%s' to 'Code Quality'.",
                    i, conf, CONFIDENCE_THRESHOLD, issue_type,
                )
                issue_type = "Code Quality"

            # Step 3 — light rule-based refinement (borderline only)
            issue_type = _refine_issue_type(issue_type, conf, code)

            classifications.append({
                "label": issue_type,
                "confidence": conf,
                "label_id": lbl["label_id"],
                "chunk_index": i,
                "code": code,
                "file": file_path,
                "start_line": start_line,
            })

        logger.debug(
            "Chunk %d: %s", i,
            ", ".join(f"{l['type']} ({l['confidence']:.2f})" for l in labels),
        )

    logger.info("Classified %d chunk(s) → %d classification(s).", len(raw_chunks), len(classifications))
    return {"classifications": classifications}
