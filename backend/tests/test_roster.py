"""Tests for roster mode: only students an admin has imported can get in, from any address."""

import re
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

from app.api.auth import COOKIE_NAME, get_google_transport
from app.auth import service
from app.config import Settings, get_settings
from app.email import get_email_sender
from app.main import create_app
from app.models import User
from tests.conftest import auth_header

V1 = "/api/v1"
CLIENT_ID = "client-123.apps.googleusercontent.com"
GMAIL = "asha.rao@gmail.com"


class SpyMailer:
    """Records sent mail."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to: str, subject: str, body: str, message_id: str | None = None) -> None:
        self.sent.append((to, subject, body))

    def last_token(self) -> str:
        found = re.search(r"token=(\S+)", self.sent[-1][2])
        assert found
        return found.group(1)


class FakeGoogle:
    """Stands in for Google's token endpoint."""

    def __init__(self) -> None:
        self.email = GMAIL

    def handler(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id_token": jwt.encode(self.claims, "k", "HS256")})

    claims: dict[str, Any] = {}


@pytest.fixture
def roster(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "registration_mode", "roster")
    monkeypatch.setattr(settings, "allowed_email_domain", "college.edu")
    monkeypatch.setattr(settings, "frontend_url", "http://app.test")


@pytest.fixture
def google_on(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "google_client_id", CLIENT_ID)
    monkeypatch.setattr(settings, "google_client_secret", "secret")
    monkeypatch.setattr(settings, "google_redirect_uri", "http://testserver/cb")


@pytest.fixture
def mailer() -> SpyMailer:
    return SpyMailer()


@pytest.fixture
def google() -> FakeGoogle:
    return FakeGoogle()


@pytest.fixture
def api(db: Session, mailer: SpyMailer, google: FakeGoogle) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_email_sender] = lambda: mailer
    app.dependency_overrides[get_google_transport] = lambda: httpx.MockTransport(google.handler)
    yield TestClient(app, follow_redirects=False)


def sign_in_with_google(api: TestClient, google: FakeGoogle, email: str) -> httpx.Response:
    """Run the Google flow for `email`."""
    location = api.get(f"{V1}/auth/google/start").headers["location"]
    sent = {key: values[0] for key, values in parse_qs(urlparse(location).query).items()}
    google.claims = {
        "iss": "https://accounts.google.com",
        "aud": CLIENT_ID,
        "sub": "g-1",
        "email": email,
        "email_verified": True,
        "nonce": sent["nonce"],
        "exp": int(time.time()) + 600,
    }
    return api.get(f"{V1}/auth/google/callback", params={"code": "c", "state": sent["state"]})


def test_the_mode_must_be_domain_or_roster() -> None:
    assert Settings().registration_mode == "domain"
    with pytest.raises(ValueError):
        Settings(registration_mode="anyone")


@pytest.mark.usefixtures("roster")
class TestRosterMode:
    def test_the_login_page_is_told_the_mode(self, api: TestClient) -> None:
        assert api.get(f"{V1}/auth/providers").json()["registration"] == "roster"

    def test_open_sign_up_is_closed_for_every_address(self, api: TestClient, db: Session) -> None:
        for email in (GMAIL, "someone@college.edu"):
            response = api.post(
                f"{V1}/auth/register",
                json={"email": email, "password": "a-long-password-1", "full_name": "Asha"},
            )
            assert response.status_code == 403
            assert "department" in response.json()["error"]["message"].lower()
        assert db.scalars(select(User)).all() == []

    def test_an_admin_can_import_personal_addresses(
        self, api: TestClient, make_user: Callable[..., User], db: Session
    ) -> None:
        admin = make_user("admin")
        csv_file = "email,full_name,roll_number,department,semester\n"
        csv_file += f"{GMAIL},Asha Rao,2201001,CSE-AI,7\n"
        response = api.post(
            f"{V1}/admin/users/import",
            headers=auth_header(admin),
            files={"file": ("students.csv", csv_file.encode(), "text/csv")},
        )
        assert response.json()["created"] == 1
        assert db.scalars(select(User).where(User.email == GMAIL)).one().department == "CSE-AI"

    def test_an_imported_student_activates_with_a_reset_link(
        self, api: TestClient, mailer: SpyMailer, make_user: Callable[..., User]
    ) -> None:
        make_user(email=GMAIL)
        api.post(f"{V1}/auth/forgot-password", json={"email": GMAIL})
        api.post(
            f"{V1}/auth/reset-password",
            json={"token": mailer.last_token(), "new_password": "a-long-password-1"},
        )
        login = api.post(f"{V1}/auth/login", json={"email": GMAIL, "password": "a-long-password-1"})
        assert login.status_code == 200

    def test_an_outsider_gets_no_reset_link(
        self, api: TestClient, mailer: SpyMailer, make_user: Callable[..., User]
    ) -> None:
        make_user(email=GMAIL)
        outsider = api.post(f"{V1}/auth/forgot-password", json={"email": "outsider@gmail.com"})
        student = api.post(f"{V1}/auth/forgot-password", json={"email": GMAIL})
        # Same reply either way, and mail only for the student on the roster.
        assert outsider.json() == student.json()
        assert [to for to, _, _ in mailer.sent] == [GMAIL]

    @pytest.mark.usefixtures("google_on")
    def test_google_signs_in_a_student_on_the_roster(
        self, api: TestClient, google: FakeGoogle, make_user: Callable[..., User]
    ) -> None:
        make_user(email=GMAIL)
        response = sign_in_with_google(api, google, GMAIL)
        assert response.headers["location"] == "http://app.test/"
        assert COOKIE_NAME in response.cookies

    @pytest.mark.usefixtures("google_on")
    def test_google_never_creates_an_account_in_roster_mode(
        self, api: TestClient, google: FakeGoogle, db: Session
    ) -> None:
        response = sign_in_with_google(api, google, "stranger@gmail.com")
        assert response.headers["location"] == "http://app.test/login?error=roster"
        assert COOKIE_NAME not in response.cookies
        assert db.scalars(select(User)).all() == []

    @pytest.mark.usefixtures("google_on")
    def test_no_domain_hint_is_sent_to_google(self, api: TestClient) -> None:
        location = api.get(f"{V1}/auth/google/start").headers["location"]
        assert "hd" not in parse_qs(urlparse(location).query)


class TestDomainModeUnchanged:
    def test_providers_report_open_registration(self, api: TestClient) -> None:
        assert api.get(f"{V1}/auth/providers").json()["registration"] == "open"

    def test_a_personal_address_still_cannot_register(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings(), "allowed_email_domain", "college.edu")
        response = api.post(
            f"{V1}/auth/register",
            json={"email": GMAIL, "password": "a-long-password-1", "full_name": "Asha"},
        )
        assert response.status_code == 422

    def test_roster_mode_accepts_any_domain_for_imports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings(), "registration_mode", "roster")
        assert service.domain_allowed(GMAIL)
