"""
Deploy the CodeSheriff inference Space to HuggingFace.

Usage (from project root):
    python scripts/deploy_inference_space.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from huggingface_hub import HfApi
from utils.config import HF_TOKEN

SPACE_ID = "jayansh21/codesheriff-inference"
SPACE_DIR = _ROOT / "spaces" / "inference"


def main():
    if not HF_TOKEN or HF_TOKEN.startswith("your_"):
        print("ERROR: Set a valid HF_TOKEN in .env first.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Create the Space repo (no-op if it already exists)
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"Space repo ready: {SPACE_ID}")

    # Upload the Space files
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
    )

    url = f"https://huggingface.co/spaces/{SPACE_ID}"
    api_url = f"https://{SPACE_ID.replace('/', '-')}.hf.space"
    print(f"✅ Space deployed: {url}")
    print(f"   API endpoint:  {api_url}")


if __name__ == "__main__":
    main()
