"""Tests for Google sign-in, with Google's token endpoint replaced by a stub."""

import json
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import COOKIE_NAME, GOOGLE_STATE_COOKIE, get_google_transport
from app.auth.passwords import hash_password
from app.config import get_settings
from app.main import create_app
from app.models import AuditLog, User
from app.models.base import utcnow

V1 = "/api/v1"
CLIENT_ID = "client-123.apps.googleusercontent.com"
REDIRECT = "http://testserver/api/v1/auth/google/callback"


class FakeGoogle:
    """Stands in for Google's token endpoint and records what it was sent."""

    def __init__(self) -> None:
        self.claims: dict[str, Any] = {}
        self.requests: list[dict[str, list[str]]] = []
        self.status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode())
        self.requests.append(form)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "invalid_grant"})
        id_token = jwt.encode(self.claims, "google-signs-this", algorithm="HS256")
        return httpx.Response(200, json={"id_token": id_token, "access_token": "unused"})


@pytest.fixture
def google_on(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "google_client_id", CLIENT_ID)
    monkeypatch.setattr(settings, "google_client_secret", "secret")
    monkeypatch.setattr(settings, "google_redirect_uri", REDIRECT)
    monkeypatch.setattr(settings, "allowed_email_domain", "college.edu")
    monkeypatch.setattr(settings, "frontend_url", "http://app.test")


@pytest.fixture
def google() -> FakeGoogle:
    return FakeGoogle()


@pytest.fixture
def api(db: Session, google: FakeGoogle) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_google_transport] = lambda: httpx.MockTransport(google.handler)
    yield TestClient(app, follow_redirects=False)


def start(api: TestClient) -> dict[str, str]:
    """Begin a sign-in and return the query sent to Google."""
    response = api.get(f"{V1}/auth/google/start")
    assert response.status_code == 302
    query = parse_qs(urlparse(response.headers["location"]).query)
    return {key: values[0] for key, values in query.items()}


def finish(api: TestClient, google: FakeGoogle, **claims: Any) -> httpx.Response:
    """Complete a sign-in as Google would, with the given ID-token claims."""
    sent = start(api)
    google.claims = {
        "iss": "https://accounts.google.com",
        "aud": CLIENT_ID,
        "sub": "google-user-1",
        "email": "asha@college.edu",
        "email_verified": True,
        "name": "Asha Rao",
        "nonce": sent["nonce"],
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        **claims,
    }
    return api.get(
        f"{V1}/auth/google/callback", params={"code": "auth-code", "state": sent["state"]}
    )


def failed_with(response: httpx.Response, reason: str) -> bool:
    """The browser was sent back to the login page with this reason and no session."""
    location = response.headers.get("location", "")
    return (
        response.status_code == 302
        and location == f"http://app.test/login?error={reason}"
        and COOKIE_NAME not in response.cookies
    )


class TestProviders:
    def test_google_is_off_until_configured(self, api: TestClient) -> None:
        assert api.get(f"{V1}/auth/providers").json() == {"google": False}
        assert api.get(f"{V1}/auth/google/start").status_code == 404

    def test_google_is_on_when_configured(self, api: TestClient, google_on: None) -> None:
        assert api.get(f"{V1}/auth/providers").json() == {"google": True}


@pytest.mark.usefixtures("google_on")
class TestStart:
    def test_redirects_to_google_with_pkce_state_and_nonce(self, api: TestClient) -> None:
        response = api.get(f"{V1}/auth/google/start")
        location = urlparse(response.headers["location"])
        query = {key: values[0] for key, values in parse_qs(location.query).items()}
        assert f"{location.scheme}://{location.netloc}" == "https://accounts.google.com"
        assert query["client_id"] == CLIENT_ID
        assert query["redirect_uri"] == REDIRECT
        assert query["response_type"] == "code"
        assert set(query["scope"].split()) == {"openid", "email", "profile"}
        assert query["code_challenge_method"] == "S256"
        assert query["hd"] == "college.edu"
        assert len(query["state"]) >= 32 and len(query["nonce"]) >= 32

    def test_state_cookie_survives_the_redirect_back_from_google(self, api: TestClient) -> None:
        response = api.get(f"{V1}/auth/google/start")
        cookie = response.headers["set-cookie"]
        assert cookie.startswith(f"{GOOGLE_STATE_COOKIE}=")
        # Strict cookies are not sent on a cross-site redirect, so the state would be lost.
        assert "samesite=lax" in cookie.lower()
        assert "httponly" in cookie.lower()

    def test_no_hosted_domain_hint_with_several_domains(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings(), "allowed_email_domain", "college.edu, dept.college.edu")
        assert "hd" not in start(api)


@pytest.mark.usefixtures("google_on")
class TestCallback:
    def test_first_sign_in_creates_a_verified_student(
        self, api: TestClient, google: FakeGoogle, db: Session
    ) -> None:
        response = finish(api, google)
        assert response.status_code == 302
        assert response.headers["location"] == "http://app.test/"
        assert COOKIE_NAME in response.cookies

        user = db.scalars(select(User).where(User.email == "asha@college.edu")).one()
        assert (user.role, user.full_name, user.password_hash) == ("student", "Asha Rao", None)
        assert user.email_verified_at is not None

        form = google.requests[0]
        assert form["code"] == ["auth-code"]
        assert form["redirect_uri"] == [REDIRECT]
        assert len(form["code_verifier"][0]) >= 43

        audit = db.scalars(select(AuditLog).where(AuditLog.action == "auth.login")).one()
        assert audit.actor_id == user.id
        assert audit.details == {"method": "google"}

    def test_the_session_works_after_sign_in(self, api: TestClient, google: FakeGoogle) -> None:
        finish(api, google)
        refreshed = api.post(f"{V1}/auth/refresh")
        assert refreshed.status_code == 200
        me = api.get(
            f"{V1}/users/me",
            headers={"Authorization": f"Bearer {refreshed.json()['access_token']}"},
        )
        assert me.json()["email"] == "asha@college.edu"

    def test_an_existing_account_is_reused(
        self,
        api: TestClient,
        google: FakeGoogle,
        db: Session,
        make_user: Callable[..., User],
    ) -> None:
        existing = make_user(
            email="asha@college.edu",
            password_hash=hash_password("a-long-password-1"),
            role="faculty",
        )
        response = finish(api, google, email="Asha@College.edu")
        assert response.headers["location"] == "http://app.test/"
        users = db.scalars(select(User).where(User.email == "asha@college.edu")).all()
        assert [user.id for user in users] == [existing.id]
        db.refresh(existing)
        assert existing.role == "faculty"
        assert existing.email_verified_at is not None

    def test_another_domain_is_refused(
        self, api: TestClient, google: FakeGoogle, db: Session
    ) -> None:
        assert failed_with(finish(api, google, email="asha@gmail.com"), "domain")
        assert db.scalars(select(User)).all() == []

    def test_an_unverified_google_address_is_refused(
        self, api: TestClient, google: FakeGoogle
    ) -> None:
        assert failed_with(finish(api, google, email_verified=False), "google")

    @pytest.mark.parametrize(
        "claims",
        [
            {"aud": "someone-else.apps.googleusercontent.com"},
            {"iss": "https://evil.example"},
            {"nonce": "replayed-nonce"},
            {"exp": int(time.time()) - 60},
        ],
    )
    def test_a_token_that_fails_any_check_is_refused(
        self, api: TestClient, google: FakeGoogle, claims: dict[str, Any]
    ) -> None:
        assert failed_with(finish(api, google, **claims), "google")

    def test_a_mismatched_state_is_refused(self, api: TestClient, google: FakeGoogle) -> None:
        start(api)
        response = api.get(
            f"{V1}/auth/google/callback", params={"code": "auth-code", "state": "forged"}
        )
        assert failed_with(response, "google")
        assert google.requests == []

    def test_a_missing_state_cookie_is_refused(self, api: TestClient, google: FakeGoogle) -> None:
        sent = start(api)
        api.cookies.clear()
        response = api.get(
            f"{V1}/auth/google/callback", params={"code": "auth-code", "state": sent["state"]}
        )
        assert failed_with(response, "google")

    def test_the_state_cannot_be_used_twice(self, api: TestClient, google: FakeGoogle) -> None:
        sent = start(api)
        assert finish_with_state(api, google, sent).headers["location"] == "http://app.test/"
        assert failed_with(finish_with_state(api, google, sent), "google")

    def test_the_user_cancelling_at_google_is_handled(self, api: TestClient) -> None:
        start(api)
        response = api.get(f"{V1}/auth/google/callback", params={"error": "access_denied"})
        assert failed_with(response, "cancelled")

    def test_a_token_endpoint_failure_is_refused(self, api: TestClient, google: FakeGoogle) -> None:
        google.status = 400
        assert failed_with(finish(api, google), "google")

    def test_a_deactivated_account_is_refused(
        self, api: TestClient, google: FakeGoogle, make_user: Callable[..., User]
    ) -> None:
        make_user(email="asha@college.edu", is_active=False, email_verified_at=utcnow())
        assert failed_with(finish(api, google), "google")

    def test_a_failure_is_audited(self, api: TestClient, google: FakeGoogle, db: Session) -> None:
        finish(api, google, email="asha@gmail.com")
        audit = db.scalars(select(AuditLog).where(AuditLog.action == "auth.login_failed")).one()
        assert audit.details is not None
        assert json.loads(json.dumps(audit.details))["method"] == "google"


def finish_with_state(api: TestClient, google: FakeGoogle, sent: dict[str, str]) -> httpx.Response:
    """Complete a sign-in that was already started."""
    google.claims = {
        "iss": "accounts.google.com",
        "aud": CLIENT_ID,
        "sub": "google-user-1",
        "email": "asha@college.edu",
        "email_verified": True,
        "nonce": sent["nonce"],
        "exp": int(time.time()) + 600,
    }
    return api.get(f"{V1}/auth/google/callback", params={"code": "c", "state": sent["state"]})
