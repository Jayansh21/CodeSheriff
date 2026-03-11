"""
Agent Node — classify_chunks

Runs the fine-tuned CodeBERT classifier on each code chunk.

The ML model is the **sole** detection engine.  No hardcoded keyword/regex
detection exists here.  Three lightweight post-processing steps refine the
raw ML output:

  1. **Confidence gate** — predictions below ``CONFIDENCE_THRESHOLD``
     (default 0.60, configurable via env) are downgraded to "Code Quality".
  2. **Confidence lock** — if ML confidence >= 0.95 the prediction is
     trusted absolutely and refinement rules are skipped.
  3. **Post-classification refinement** — generic code-pattern rules
     can *adjust* the issue type for moderate-confidence predictions.
     They never *replace* the ML classifier for strong predictions.
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


# Reverse map: user-facing issue type → representative label_id
# Used when refinement upgrades a Clean chunk (label_id 0) to an issue.
_ISSUE_TYPE_TO_LABEL_ID = {
    "Security Vulnerability": 3,
    "Runtime Bug": 1,
    "Type Bug": 2,
    "Logic Bug": 4,
    "Code Quality": 0,
    "Clean": 0,
}


# ---------------------------------------------------------------------------
# Trivial-chunk filter
# Skips chunks that contain no meaningful logic (imports, bare declarations,
# blank lines, etc.) to avoid false positives from non-functional fragments.
# ---------------------------------------------------------------------------
_IMPORT_RE = re.compile(r"^\s*(?:import\s|from\s\S+\s+import\s)")
_BARE_DECL_RE = re.compile(r"^\s*(?:class\s+\w+|def\s+\w+|async\s+def\s+\w+)")
_ASSIGNMENT_RE = re.compile(r"^\s*self\.\w+\s*=")
_PASS_DOCSTRING_RE = re.compile(r'^\s*(?:pass|"""|\'\'\'|#)')


def _is_trivial_chunk(code: str) -> bool:
    """
    Return True if the chunk contains no meaningful logic worth classifying.

    Trivial chunks include:
      - Import-only blocks
      - Bare class/function declarations without a body
      - Chunks with only assignments, pass, docstrings, or comments
    """
    lines = [ln for ln in code.splitlines() if ln.strip()]
    if not lines:
        return True

    # Count how many lines are "structural" vs "logic"
    structural = 0
    for ln in lines:
        stripped = ln.strip()
        if (_IMPORT_RE.match(stripped)
                or _BARE_DECL_RE.match(stripped)
                or _PASS_DOCSTRING_RE.match(stripped)
                or stripped == ""):
            structural += 1

    logic_lines = len(lines) - structural

    # If there are no logic lines, it's trivial
    if logic_lines == 0:
        return True

    # A single assignment (e.g. self.db = ...) with no other logic is trivial
    if logic_lines == 1:
        for ln in lines:
            stripped = ln.strip()
            if (not _IMPORT_RE.match(stripped)
                    and not _BARE_DECL_RE.match(stripped)
                    and not _PASS_DOCSTRING_RE.match(stripped)
                    and _ASSIGNMENT_RE.match(stripped)):
                return True

    return False


# ---------------------------------------------------------------------------
# Post-classification refinement rules.
#
# Two tiers:
#   HIGH-SIGNAL — patterns the ML model consistently misclassifies as Clean.
#                 These run when ML confidence is moderate, because the model
#                 has known blind spots for them.
#   BORDERLINE  — softer adjustments that only fire when ML confidence is
#                 below _BORDERLINE_CEILING.
#
# GUARD: If ML confidence >= 0.95, refinement rules are **never** applied.
# This ensures very strong ML predictions are always trusted.
#
# All rules *adjust* the issue type after ML runs — they never bypass it.
# ---------------------------------------------------------------------------
_CONFIDENCE_LOCK = 0.95     # above this, refinement rules are skipped entirely
_BORDERLINE_CEILING = 0.75  # borderline rules only fire below this

# Compiled once — each tuple is (pattern, target_issue_type).
_HIGH_SIGNAL_RULES = [
    # `if x == val or "literal":` — always-True logic bug
    (re.compile(r'\bif\b.+\bor\s+["\'][^"\']+["\']'), "Logic Bug"),
    # `if x == val or var:` without comparison — truthiness bug
    (re.compile(r"\bif\s+\w+\s*==\s*.+?\bor\s+\w+\s*:"), "Logic Bug"),
    # Division by len() of possibly-empty collection
    (re.compile(r"/\s*len\("), "Runtime Bug"),
    # Unguarded .fetchone()[index]
    (re.compile(r"\.fetchone\(\)\s*\["), "Runtime Bug"),
    # Unguarded .fetchone().attr
    (re.compile(r"\.fetchone\(\)\.\w+"), "Runtime Bug"),
]

_BORDERLINE_RULES = [
    # subprocess / os.system with shell=True or string concat
    (re.compile(r"\bsubprocess\.\w+\(.+shell\s*=\s*True", re.DOTALL), "Security Vulnerability"),
    (re.compile(r"\bos\.system\s*\("), "Security Vulnerability"),
]


def _refine_issue_type(issue_type: str, confidence: float, code: str) -> str:
    """
    Adjust *issue_type* based on well-known code patterns the ML model
    has blind spots for.

    Guard: if ML confidence >= 0.95 the model is very sure — refinement
    rules are skipped entirely so strong ML predictions are never overridden.

    High-signal rules run for moderate confidence.  Borderline rules only
    fire when confidence < 0.75.
    """
    # Very confident ML prediction — trust it, skip all rules
    if confidence >= _CONFIDENCE_LOCK:
        return issue_type

    # High-signal rules — apply for moderate confidence
    for pattern, target in _HIGH_SIGNAL_RULES:
        if pattern.search(code):
            return target

    # Borderline rules — only when ML is uncertain
    if confidence < _BORDERLINE_CEILING:
        for pattern, target in _BORDERLINE_RULES:
            if pattern.search(code):
                return target

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

        # Skip trivial chunks (imports, bare declarations, etc.)
        if _is_trivial_chunk(code):
            logger.debug("Chunk %d skipped (trivial — no logic).", i)
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
            label_id = lbl["label_id"]

            # Step 1 — resolve model label to user-facing issue type
            issue_type = _resolve_issue_type(raw_type)

            # Step 2 — confidence gate: weak predictions become Code Quality
            if conf < CONFIDENCE_THRESHOLD and issue_type not in ("Clean", "Code Quality"):
                logger.debug(
                    "Chunk %d: confidence %.2f < %.2f — downgrading '%s' to 'Code Quality'.",
                    i, conf, CONFIDENCE_THRESHOLD, issue_type,
                )
                issue_type = "Code Quality"

            # Step 3 — pattern-based refinement (may upgrade Clean/Code Quality)
            refined = _refine_issue_type(issue_type, conf, code)

            # If refinement changed the issue type, update label_id so
            # prioritize_issues doesn't filter it as Clean (label_id 0).
            if refined != issue_type:
                logger.debug(
                    "Chunk %d: refinement '%s' -> '%s'.", i, issue_type, refined,
                )
                issue_type = refined
                if label_id == 0:
                    label_id = _ISSUE_TYPE_TO_LABEL_ID.get(issue_type, 4)

            classifications.append({
                "label": issue_type,
                "confidence": conf,
                "label_id": label_id,
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
