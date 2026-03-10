"""
CodeSheriff — FastAPI Backend

Endpoints:
    GET  /health      → service liveness + model status
    GET  /ping        → lightweight uptime check
    POST /test-diff   → run the full review pipeline on a sample diff
    POST /review      → run the pipeline on a user-supplied diff
    POST /webhook     → GitHub App webhook receiver (async processing)

Usage (from project root):
    uvicorn backend.main:app --reload
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import SAMPLE_DIFF, MODELS_DIR
from utils.logger import get_logger
from agents.graph import run_review

logger = get_logger("backend.main")

# Track model availability
_model_available = False


# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_available
    try:
        from ml.inference import predict_bug  # noqa: F401 — triggers model load
        _model_available = True
        logger.info("Model loaded on startup successfully")
    except Exception as e:
        logger.error(f"Model failed to load on startup: {e}")
        logger.warning("Pipeline will use heuristic classifier.")
    logger.info("CodeSheriff API started ✅")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CodeSheriff",
    description="AI-powered GitHub PR reviewer",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DiffRequest(BaseModel):
    diff: str = ""


class ReviewResponse(BaseModel):
    issues_found: int
    review: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model_available,
    }


@app.get("/ping")
async def ping():
    """Lightweight endpoint for uptime monitoring (e.g. cron pinger)."""
    return {"ping": "pong"}


@app.post("/test-diff", response_model=ReviewResponse)
async def test_diff():
    """Run the pipeline on the built-in sample diff."""
    try:
        review = run_review(SAMPLE_DIFF)
        issue_count = review.count("## Issue")
        return ReviewResponse(issues_found=issue_count, review=review)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review", response_model=ReviewResponse)
async def review_diff(request: DiffRequest):
    """Run the pipeline on a user-supplied diff."""
    diff = request.diff.strip()
    if not diff:
        raise HTTPException(status_code=400, detail="Diff body is empty.")
    try:
        review = run_review(diff)
        issue_count = review.count("## Issue")
        return ReviewResponse(issues_found=issue_count, review=review)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GitHub Webhook — responds immediately, processes in background
# ---------------------------------------------------------------------------
async def _process_webhook(payload: dict):
    """Background task: run the review pipeline on a PR webhook event."""
    try:
        action = payload.get("action", "")
        if action not in ("opened", "synchronize", "reopened"):
            logger.info(f"Ignoring webhook action: {action}")
            return

        pr = payload.get("pull_request", {})
        diff_url = pr.get("diff_url", "")
        if not diff_url:
            logger.warning("No diff_url in webhook payload")
            return

        # Fetch the diff from GitHub
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(diff_url, timeout=30.0)
            resp.raise_for_status()
            diff_text = resp.text

        review = run_review(diff_text)

        # Post review comment back to the PR
        from backend.github_auth import get_installation_token
        installation_id = payload.get("installation", {}).get("id")
        if not installation_id:
            logger.warning("No installation ID — cannot post review comment")
            return

        token = await get_installation_token(installation_id)
        repo_full_name = payload.get("repository", {}).get("full_name", "")
        pr_number = pr.get("number")

        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                json={"body": review},
                timeout=30.0,
            )
        logger.info(f"Posted review to {repo_full_name}#{pr_number}")

    except Exception as e:
        logger.error(f"Background webhook processing failed: {e}", exc_info=True)


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    GitHub App webhook receiver.
    Returns 200 immediately so GitHub never times out (even on cold start).
    Actual review processing happens in a background task.
    """
    from backend.github_auth import verify_webhook_signature

    body = await request.body()
    signature = request.headers.get("x-hub-signature-256", "")

    if not verify_webhook_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    background_tasks.add_task(_process_webhook, payload)
    return {"status": "received"}
