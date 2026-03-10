"""
CodeSheriff — GitHub App Authentication

Handles JWT generation, installation token exchange, and webhook
signature verification for GitHub App integration.
"""

import hashlib
import hmac
import time

import httpx
import jwt

from utils.config import GITHUB_APP_ID, GITHUB_PRIVATE_KEY, GITHUB_WEBHOOK_SECRET
from utils.logger import get_logger

logger = get_logger("backend.github_auth")


def generate_jwt() -> str:
    """Generate a JWT for GitHub App authentication. Valid for 10 minutes."""
    if not GITHUB_APP_ID or not GITHUB_PRIVATE_KEY:
        raise EnvironmentError(
            "GITHUB_APP_ID and GITHUB_PRIVATE_KEY must be set. "
            "See .env.example for details."
        )
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (10 * 60),
        "iss": GITHUB_APP_ID,
    }
    return jwt.encode(payload, GITHUB_PRIVATE_KEY, algorithm="RS256")


async def get_installation_token(installation_id: int) -> str:
    """Exchange JWT for an installation access token scoped to one repo."""
    jwt_token = generate_jwt()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()["token"]


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify the webhook came from GitHub using HMAC-SHA256."""
    if not GITHUB_WEBHOOK_SECRET:
        logger.warning("GITHUB_WEBHOOK_SECRET not set — skipping verification")
        return True
    expected = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
