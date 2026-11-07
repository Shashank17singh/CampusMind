"""Google sign-in with OpenID Connect: authorization code flow with PKCE, state and nonce.

The browser goes to Google with a random state and nonce, and a PKCE challenge. Their
values travel in a signed, short-lived cookie, so nothing is stored on the server. Google
sends the browser back with a code, which the server exchanges for an ID token.

The ID token comes straight from Google's token endpoint over TLS, so its signature is not
checked again. OpenID Connect Core 3.1.3.7 allows TLS to stand in for the signature in
this case. Its issuer, audience, expiry and nonce are still checked.
"""

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt

from app.auth.tokens import InvalidTokenError, _decode, _encode
from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105, a URL, not a password
ISSUERS = {"accounts.google.com", "https://accounts.google.com"}
STATE_MINUTES = 10


class GoogleSignInError(Exception):
    """Google sign-in could not be completed. `reason` is safe to show in a URL."""

    def __init__(self, message: str, reason: str = "google") -> None:
        """Keep the reason the browser is told about, separate from the logged message."""
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class GoogleIdentity:
    """Who Google says signed in."""

    subject: str
    email: str
    name: str


def _challenge(verifier: str) -> str:
    """The S256 PKCE challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def begin() -> tuple[str, str]:
    """Return the URL to send the browser to, and the signed state cookie value."""
    settings = get_settings()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": _challenge(verifier),
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }
    domains = settings.allowed_email_domains
    if len(domains) == 1:
        # A hint that narrows Google's account chooser. The email check below still decides.
        params["hd"] = next(iter(domains))
    cookie = _encode(
        {"type": "google_state", "state": state, "nonce": nonce, "verifier": verifier},
        timedelta(minutes=STATE_MINUTES),
    )
    return f"{AUTHORIZATION_URL}?{urlencode(params)}", cookie


def _read_state(cookie: str | None, state: str | None) -> dict[str, Any]:
    """Check the returned state against the signed cookie, and return the cookie's claims."""
    if not cookie or not state:
        raise GoogleSignInError("missing state")
    try:
        claims = _decode(cookie, "google_state")
    except InvalidTokenError as exc:
        raise GoogleSignInError(f"invalid state cookie: {exc}") from exc
    if not secrets.compare_digest(str(claims.get("state", "")), state):
        raise GoogleSignInError("state mismatch")
    return claims


def _exchange(code: str, verifier: str, transport: httpx.BaseTransport | None) -> str:
    """Swap the authorization code for an ID token."""
    settings = get_settings()
    form = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": settings.google_redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": verifier,
    }
    try:
        with httpx.Client(timeout=15, transport=transport) as client:
            response = client.post(TOKEN_URL, data=form)
        response.raise_for_status()
        id_token = response.json()["id_token"]
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise GoogleSignInError(f"token exchange failed: {exc}") from exc
    return str(id_token)


def _check_id_token(id_token: str, nonce: str) -> dict[str, Any]:
    """Validate the ID token's claims. The signature is covered by TLS, as explained above."""
    settings = get_settings()
    try:
        claims: dict[str, Any] = jwt.decode(
            id_token,
            # PyJWT skips every other check when the signature check is off, unless each is
            # switched back on here.
            options={
                "verify_signature": False,
                "verify_aud": True,
                "verify_exp": True,
                "require": ["iss", "aud", "exp", "sub"],
            },
            audience=settings.google_client_id,
            algorithms=["RS256", "HS256"],
        )
    except jwt.PyJWTError as exc:
        raise GoogleSignInError(f"invalid ID token: {exc}") from exc
    if claims.get("iss") not in ISSUERS:
        raise GoogleSignInError("wrong issuer")
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise GoogleSignInError("nonce mismatch")
    if claims.get("email_verified") is not True or not claims.get("email"):
        raise GoogleSignInError("email not verified by Google")
    return claims


def complete(
    *,
    code: str | None,
    state: str | None,
    cookie: str | None,
    error: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> GoogleIdentity:
    """Finish a sign-in from the values Google sent back, or raise GoogleSignInError."""
    if error:
        raise GoogleSignInError(f"Google returned {error}", reason="cancelled")
    claims = _read_state(cookie, state)
    if not code:
        raise GoogleSignInError("missing code")
    token_claims = _check_id_token(_exchange(code, claims["verifier"], transport), claims["nonce"])
    return GoogleIdentity(
        subject=str(token_claims["sub"]),
        email=str(token_claims["email"]),
        name=str(token_claims.get("name") or token_claims["email"].split("@")[0]),
    )
