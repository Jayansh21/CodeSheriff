"""
CodeSheriff Configuration Module
Loads and validates environment variables using python-dotenv.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Determine project root as the directory containing this file's parent
# (i.e., CodeSheriff/).  This allows the module to be imported from anywhere.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root (no-op if the file doesn't exist)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Global reproducibility seed — used in dataset splitting, training, etc.
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
MODEL_PATH: str = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "codesheriff-model" / "final"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
USE_LOCAL_MODEL: bool = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
INFERENCE_SPACE_ID: str = os.getenv("INFERENCE_SPACE_ID", "jayansh21/codesheriff-inference")

# GitHub App authentication
GITHUB_APP_ID: str = os.getenv("GITHUB_APP_ID", "")
GITHUB_PRIVATE_KEY: str = os.getenv("GITHUB_PRIVATE_KEY", "").replace("\\n", "\n")
GITHUB_WEBHOOK_SECRET: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "codesheriff-model"

# Ensure critical directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Label mapping for the bug classifier
# ---------------------------------------------------------------------------
LABEL_NAMES = {
    0: "Clean",
    1: "Null Reference Risk",
    2: "Type Mismatch",
    3: "Security Vulnerability",
    4: "Logic Flaw",
}
NUM_LABELS = len(LABEL_NAMES)

# ---------------------------------------------------------------------------
# Model / training constants
# ---------------------------------------------------------------------------
MAX_TOKEN_LENGTH = 512
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
MAX_DATASET_SAMPLES = 10_000

# ---------------------------------------------------------------------------
# Groq LLM settings
# ---------------------------------------------------------------------------
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_MAX_RETRIES = 3
GROQ_TIMEOUT_SECONDS = 10


def require_groq_key() -> str:
    """Return the Groq API key or raise a clear error."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and fill in your Groq API key."
        )
    return GROQ_API_KEY


# ---------------------------------------------------------------------------
# Sample test diff used throughout demos, tests, and API validation
# ---------------------------------------------------------------------------
SAMPLE_DIFF = r"""
--- a/app/user_service.py
+++ b/app/user_service.py
@@ -1,20 +1,30 @@
 import sqlite3

 def get_user(user_id):
-    query = "SELECT * FROM users WHERE id = " + user_id
-    conn = sqlite3.connect("users.db")
-    result = conn.execute(query)
-    return result.fetchone().name
+    query = "SELECT * FROM users WHERE id = ?"
+    conn = sqlite3.connect("users.db")
+    result = conn.execute(query, (user_id,))
+    row = result.fetchone()
+    return row.name if row else None

 def calculate_discount(price, discount):
-    if discount = 100:
+    if discount == 100:
         return 0
-    return price / discount
+    return price * (1 - discount / 100)

 def process_items(items):
     total = 0
-    for i in range(len(items) + 1):
+    for i in range(len(items)):
         total += items[i]
     return total

 def load_config(path):
     config = None
-    print(config.settings)
+    if config is not None:
+        print(config.settings)
""".strip()
