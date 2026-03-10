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

import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("backend.main")

# ---------------------------------------------------------------------------
# Rate limiter — keyed by client IP
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# App — no model loading at startup (lazy-loads on first inference request)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CodeSheriff",
    description="AI-powered GitHub PR reviewer",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s → %d (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the CodeSheriff landing page."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodeSheriff — AI-Powered PR Reviewer</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#0d1117;color:#c9d1d9;min-height:100vh;display:flex;flex-direction:column;align-items:center}
  .hero{text-align:center;padding:80px 24px 40px}
  .hero h1{font-size:3rem;color:#58a6ff;margin-bottom:12px}
  .hero .badge{font-size:1.1rem;color:#8b949e;display:inline-block;
               background:#161b22;padding:6px 16px;border-radius:20px;border:1px solid #30363d;margin-bottom:24px}
  .hero p{font-size:1.25rem;max-width:640px;margin:0 auto 32px;line-height:1.6;color:#8b949e}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;
         max-width:800px;width:100%;padding:0 24px;margin-bottom:48px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px}
  .card h3{color:#58a6ff;margin-bottom:8px;font-size:1rem}
  .card p{color:#8b949e;font-size:0.9rem;line-height:1.5}
  .cta{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-bottom:60px}
  .cta a{text-decoration:none;padding:12px 28px;border-radius:8px;font-weight:600;font-size:1rem;transition:.2s}
  .btn-primary{background:#238636;color:#fff;border:1px solid #238636}
  .btn-primary:hover{background:#2ea043}
  .btn-secondary{background:transparent;color:#58a6ff;border:1px solid #30363d}
  .btn-secondary:hover{background:#161b22}
  footer{margin-top:auto;padding:24px;color:#484f58;font-size:0.85rem;text-align:center;
         border-top:1px solid #21262d;width:100%}
  footer a{color:#58a6ff;text-decoration:none}
</style>
</head>
<body>
  <div class="hero">
    <h1>&#x1f6e1;&#xfe0f; CodeSheriff</h1>
    <span class="badge">AI-Powered GitHub PR Reviewer</span>
    <p>CodeSheriff uses a fine-tuned CodeBERT model to classify code issues and
       an LLM agent pipeline to generate actionable fix suggestions — posted
       directly as PR comments.</p>
  </div>
  <div class="cards">
    <div class="card">
      <h3>&#x1f50d; Bug Classification</h3>
      <p>Fine-tuned CodeBERT detects security vulnerabilities, null reference risks,
         type mismatches, and logic flaws.</p>
    </div>
    <div class="card">
      <h3>&#x1f916; Agent Pipeline</h3>
      <p>LangGraph orchestrates 5 nodes: parse diff, classify, prioritise,
         generate fixes (Groq LLM), and format the review.</p>
    </div>
    <div class="card">
      <h3>&#x1f4ac; PR Comments</h3>
      <p>Install the GitHub App and CodeSheriff posts a detailed review comment
         on every pull request automatically.</p>
    </div>
  </div>
  <div class="cta">
    <a class="btn-primary" href="https://github.com/Jayansh21/CodeSheriff" target="_blank">View on GitHub</a>
    <a class="btn-secondary" href="/docs">API Docs</a>
    <a class="btn-secondary" href="/health">Health Check</a>
  </div>
  <footer>
    Built by <a href="https://github.com/Jayansh21" target="_blank">Jayansh21</a>
    &middot; Powered by CodeBERT + Groq + LangGraph
  </footer>
</body>
</html>"""


@app.get("/health")
async def health():
    try:
        from ml.inference import USE_LOCAL_MODEL, _model, _hf_client
        inference_mode = "local" if USE_LOCAL_MODEL else "hf_space"
        if USE_LOCAL_MODEL:
            model_ready = _model is not None
        else:
            model_ready = _hf_client is not None
    except Exception:
        inference_mode = "unknown"
        model_ready = False
    return {
        "status": "ok",
        "inference_mode": inference_mode,
        "model_ready": model_ready,
    }


@app.get("/ping")
async def ping():
    """Lightweight endpoint for uptime monitoring (e.g. cron pinger)."""
    return {"ping": "pong"}


@app.post("/test-diff", response_model=ReviewResponse)
@limiter.limit("5/minute")
async def test_diff(request: Request):
    """Run the pipeline on the built-in sample diff."""
    try:
        from agents.graph import run_review
        from utils.config import SAMPLE_DIFF
        review = run_review(SAMPLE_DIFF)
        issue_count = review.count("## Issue")
        return ReviewResponse(issues_found=issue_count, review=review)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review", response_model=ReviewResponse)
@limiter.limit("10/minute")
async def review_diff(request: Request, body: DiffRequest):
    """Run the pipeline on a user-supplied diff."""
    diff = body.diff.strip()
    if not diff:
        raise HTTPException(status_code=400, detail="Diff body is empty.")
    try:
        from agents.graph import run_review
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
    action = payload.get("action", "")
    repo_full_name = payload.get("repository", {}).get("full_name", "unknown")
    pr = payload.get("pull_request", {})
    pr_number = pr.get("number", "?")

    logger.info("Processing webhook: %s %s#%s", action, repo_full_name, pr_number)

    try:
        if action not in ("opened", "synchronize", "reopened"):
            logger.info("Ignoring webhook action: %s", action)
            return

        # Authenticate with GitHub API (needed for private repos)
        from backend.github_auth import get_installation_token
        import httpx

        installation_id = payload.get("installation", {}).get("id")
        if not installation_id:
            logger.warning("No installation ID — cannot fetch diff or post comment for %s#%s", repo_full_name, pr_number)
            return

        token = await get_installation_token(installation_id)

        # Fetch the diff via GitHub REST API (works for both public and private repos)
        diff_api_url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
        logger.info("Fetching diff from %s", diff_api_url)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                diff_api_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3.diff",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            diff_text = resp.text
        logger.info("Fetched diff (%d chars) for %s#%s", len(diff_text), repo_full_name, pr_number)

        from agents.graph import run_review
        review = run_review(diff_text)
        logger.info("Review generated (%d chars) for %s#%s", len(review), repo_full_name, pr_number)

        async with httpx.AsyncClient() as client:
            post_resp = await client.post(
                f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                json={"body": review},
                timeout=30.0,
            )
            post_resp.raise_for_status()
        logger.info("Posted review to %s#%s", repo_full_name, pr_number)

    except Exception as e:
        logger.error("Background webhook processing failed for %s#%s: %s", repo_full_name, pr_number, e, exc_info=True)


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    GitHub App webhook receiver.
    Returns 200 immediately so GitHub never times out (even on cold start).
    Actual review processing happens in a background task.
    """
    body = await request.body()
    payload = json.loads(body)

    # Log signature for debugging (non-blocking — never reject)
    signature = request.headers.get("x-hub-signature-256", "")
    event_type = request.headers.get("x-github-event", "unknown")
    action = payload.get("action", "")
    repo = payload.get("repository", {}).get("full_name", "unknown")
    pr_number = payload.get("pull_request", {}).get("number", "?")

    logger.info(
        "Webhook received: event=%s action=%s repo=%s PR=#%s sig=%s",
        event_type, action, repo, pr_number,
        "present" if signature else "MISSING",
    )

    background_tasks.add_task(_process_webhook, payload)
    return {"status": "received"}


# ---------------------------------------------------------------------------
# Webhook debug endpoint (temporary — remove after confirming delivery)
# ---------------------------------------------------------------------------
@app.post("/webhook-test")
async def webhook_test(request: Request):
    """Lightweight endpoint to confirm POST routing works on Render."""
    payload = await request.json()
    logger.info("Webhook test received: %s", payload)
    return {"received": True, "keys": list(payload.keys())}
