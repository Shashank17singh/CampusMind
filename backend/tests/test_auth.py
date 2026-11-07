"""Tests for accounts, sessions, rate limits and the permission matrix."""

import re
import uuid
from collections.abc import Callable, Iterator
from datetime import timedelta

import fakeredis
import jwt
import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.runner import get_agent_runner
from app.api.auth import COOKIE_NAME, get_login_limiter
from app.auth import service
from app.auth.passwords import hash_password, needs_rehash, verify_password
from app.auth.permissions import can_manage_document
from app.auth.ratelimit import LoginLimiter
from app.auth.tokens import (
    InvalidTokenError,
    create_access_token,
    create_email_verification_token,
    create_password_reset_token,
    decode_access_token,
    new_refresh_token,
    read_email_verification_token,
)
from app.config import Settings, get_settings
from app.email import get_email_sender
from app.llm import get_llm
from app.main import create_app
from app.models import AuditLog, Document, RefreshToken, User
from app.models.base import utcnow
from app.rag.cache import AnswerCache
from app.rag.service import get_rag_pipeline
from app.vectorstore import get_vector_store
from app.vectorstore.embeddings import get_embedder
from tests.conftest import auth_header
from tests.helpers import FakeLLM, HashingEmbedder
from tests.test_vectorstore import add_ready_document

PASSWORD = "correct horse battery"
NEW_PASSWORD = "a brand new password"
EMAIL = "asha@college.edu"


class SpyMailer:
    """Keeps sent messages, and can read tokens out of the links."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append((to, subject, body))

    def last_token(self) -> str:
        found = re.search(r"token=(\S+)", self.sent[-1][2])
        assert found is not None
        return found.group(1)


@pytest.fixture
def mailer() -> SpyMailer:
    return SpyMailer()


@pytest.fixture
def limiter() -> LoginLimiter:
    return LoginLimiter(fakeredis.FakeRedis(decode_responses=True))


@pytest.fixture
def api(db: Session, mailer: SpyMailer, limiter: LoginLimiter) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_email_sender] = lambda: mailer
    app.dependency_overrides[get_login_limiter] = lambda: limiter
    yield TestClient(app)


def verified_user(make_user: Callable[..., User], **fields: object) -> User:
    """Create a user who can log in."""
    defaults: dict[str, object] = {
        "email": EMAIL,
        "password_hash": hash_password(PASSWORD),
        "email_verified_at": utcnow(),
    }
    return make_user(**{**defaults, **fields})


def log_in(client: TestClient, email: str = EMAIL, password: str = PASSWORD):
    """Post to the login endpoint."""
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


class TestPasswords:
    def test_round_trip(self) -> None:
        stored = hash_password(PASSWORD)
        assert stored.startswith("$argon2")
        assert verify_password(stored, PASSWORD)
        assert not verify_password(stored, "wrong password")

    def test_hashes_are_salted(self) -> None:
        assert hash_password(PASSWORD) != hash_password(PASSWORD)

    def test_a_missing_hash_never_verifies(self) -> None:
        assert not verify_password(None, "campusmind-timing-equaliser")

    def test_a_garbage_hash_does_not_crash(self) -> None:
        assert not verify_password("not-a-hash", PASSWORD)

    def test_a_current_hash_needs_no_rehash(self) -> None:
        assert not needs_rehash(hash_password(PASSWORD))


class TestTokens:
    def test_access_token_round_trip(self) -> None:
        user_id = uuid.uuid4()
        claims = decode_access_token(create_access_token(user_id, "faculty"))
        assert (claims["sub"], claims["role"]) == (str(user_id), "faculty")

    def test_access_token_lasts_fifteen_minutes(self) -> None:
        claims = decode_access_token(create_access_token(uuid.uuid4(), "student"))
        assert claims["exp"] - claims["iat"] == 900

    def test_expired_token_is_rejected(self) -> None:
        settings = get_settings()
        token = jwt.encode(
            {"sub": "x", "type": "access", "exp": utcnow() - timedelta(minutes=1)},
            settings.secret_key,
            algorithm="HS256",
        )
        with pytest.raises(InvalidTokenError):
            decode_access_token(token)

    def test_tampered_token_is_rejected(self) -> None:
        token = create_access_token(uuid.uuid4(), "student")
        with pytest.raises(InvalidTokenError):
            decode_access_token(token[:-2] + "xx")

    def test_a_token_signed_with_another_key_is_rejected(self) -> None:
        forged = jwt.encode({"sub": "x", "type": "access"}, "k" * 40, algorithm="HS256")
        with pytest.raises(InvalidTokenError):
            decode_access_token(forged)

    def test_an_unsigned_token_is_rejected(self) -> None:
        forged = jwt.encode({"sub": "x", "type": "access"}, "", algorithm="none")
        with pytest.raises(InvalidTokenError):
            decode_access_token(forged)

    def test_an_email_token_cannot_be_used_as_an_access_token(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_access_token(create_email_verification_token(uuid.uuid4()))

    def test_an_access_token_cannot_verify_an_email(self) -> None:
        with pytest.raises(InvalidTokenError):
            read_email_verification_token(create_access_token(uuid.uuid4(), "student"))

    def test_verification_token_lasts_24_hours(self) -> None:
        token = create_email_verification_token(uuid.uuid4())
        claims = jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
        assert claims["exp"] - claims["iat"] == 86400

    def test_refresh_tokens_are_unique_and_only_the_hash_is_kept(self) -> None:
        first, first_hash = new_refresh_token()
        second, _ = new_refresh_token()
        assert first != second
        assert first not in first_hash


class TestRegistration:
    def body(self, **fields: object) -> dict:
        return {"email": EMAIL, "password": PASSWORD, "full_name": "Asha", **fields}

    def test_register_sends_a_verification_link(
        self, api: TestClient, db: Session, mailer: SpyMailer
    ) -> None:
        response = api.post("/api/v1/auth/register", json=self.body())
        assert response.status_code == 202
        assert mailer.sent[0][0] == EMAIL
        user = db.scalar(select(User).where(User.email == EMAIL))
        assert user is not None
        assert user.role == "student"
        assert user.password_hash != PASSWORD

    def test_other_domains_are_refused(self, api: TestClient, mailer: SpyMailer) -> None:
        response = api.post("/api/v1/auth/register", json=self.body(email="asha@gmail.com"))
        assert response.status_code == 422
        assert "college email" in response.json()["error"]["message"]
        assert mailer.sent == []

    def test_a_lookalike_domain_is_refused(self, api: TestClient) -> None:
        response = api.post(
            "/api/v1/auth/register", json=self.body(email="asha@college.edu.evil.com")
        )
        assert response.status_code == 422

    def test_a_taken_address_gets_the_same_reply(
        self, api: TestClient, make_user: Callable[..., User], mailer: SpyMailer
    ) -> None:
        verified_user(make_user)
        response = api.post("/api/v1/auth/register", json=self.body())
        assert response.status_code == 202
        assert mailer.sent == []

    def test_case_does_not_make_a_second_account(self, api: TestClient, db: Session) -> None:
        api.post("/api/v1/auth/register", json=self.body())
        api.post("/api/v1/auth/register", json=self.body(email="ASHA@college.edu"))
        assert db.scalar(select(func.count()).select_from(User)) == 1

    def test_a_short_password_is_refused(self, api: TestClient) -> None:
        response = api.post("/api/v1/auth/register", json=self.body(password="short"))
        assert response.status_code == 422

    def test_nobody_can_register_as_admin(self, api: TestClient, db: Session) -> None:
        api.post("/api/v1/auth/register", json=self.body(role="admin"))
        user = db.scalar(select(User).where(User.email == EMAIL))
        assert user is not None and user.role == "student"

    def test_several_domains_can_be_allowed(self) -> None:
        settings = Settings(allowed_email_domain="college.edu, Dept.College.edu")
        assert settings.allowed_email_domains == {"college.edu", "dept.college.edu"}


class TestVerification:
    def test_the_link_verifies_the_account(
        self, api: TestClient, db: Session, mailer: SpyMailer
    ) -> None:
        api.post(
            "/api/v1/auth/register",
            json={"email": EMAIL, "password": PASSWORD, "full_name": "Asha"},
        )
        response = api.post("/api/v1/auth/verify-email", json={"token": mailer.last_token()})
        assert response.status_code == 200
        db.expire_all()
        user = db.scalar(select(User).where(User.email == EMAIL))
        assert user is not None and user.email_verified_at is not None

    def test_a_bad_token_is_refused(self, api: TestClient) -> None:
        response = api.post("/api/v1/auth/verify-email", json={"token": "nonsense"})
        assert response.status_code == 401

    def test_resend_only_goes_to_unverified_accounts(
        self, api: TestClient, make_user: Callable[..., User], mailer: SpyMailer
    ) -> None:
        verified_user(make_user)
        make_user(email="new@college.edu")
        api.post("/api/v1/auth/resend-verification", json={"email": EMAIL})
        assert mailer.sent == []
        api.post("/api/v1/auth/resend-verification", json={"email": "new@college.edu"})
        assert len(mailer.sent) == 1


class TestLogin:
    def test_success_returns_a_token_and_a_secure_cookie(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        response = log_in(api)
        assert response.status_code == 200
        body = response.json()
        assert body["expires_in"] == 900
        assert body["user"]["email"] == EMAIL
        assert "password_hash" not in body["user"]

        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "path=/api/v1/auth" in cookie

    def test_the_token_works(self, api: TestClient, make_user: Callable[..., User]) -> None:
        verified_user(make_user)
        token = log_in(api).json()["access_token"]
        me = api.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
        assert me.json()["email"] == EMAIL

    def test_a_wrong_password_and_an_unknown_user_look_the_same(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        wrong = log_in(api, password="wrong password here")
        unknown = log_in(api, email="ghost@college.edu")
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"]

    def test_an_unverified_account_is_told_to_verify(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user, email_verified_at=None)
        response = log_in(api)
        assert response.status_code == 403
        assert "verify" in response.json()["error"]["message"]

    def test_a_deactivated_account_cannot_log_in(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user, is_active=False)
        assert log_in(api).status_code == 401

    def test_an_imported_account_with_no_password_cannot_log_in(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user, password_hash=None)
        assert log_in(api).status_code == 401

    def test_login_is_case_insensitive(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        assert log_in(api, email="ASHA@College.edu").status_code == 200

    def test_success_and_failure_are_audited(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = verified_user(make_user)
        log_in(api)
        log_in(api, password="wrong password here")
        actions = db.scalars(select(AuditLog.action).order_by(AuditLog.id)).all()
        assert actions == ["auth.login", "auth.login_failed"]
        entry = db.scalar(select(AuditLog).where(AuditLog.action == "auth.login"))
        assert entry is not None and entry.actor_id == user.id

    def test_a_wrong_password_never_appears_in_the_audit_log(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        log_in(api, password="my-secret-attempt")
        entry = db.scalar(select(AuditLog).where(AuditLog.action == "auth.login_failed"))
        assert entry is not None
        assert "my-secret-attempt" not in str(entry.details)


class TestRateLimiting:
    def test_the_sixth_attempt_is_blocked_even_with_the_right_password(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        for _ in range(5):
            assert log_in(api, password="wrong password here").status_code == 401
        assert log_in(api).status_code == 429

    def test_attempts_on_many_accounts_from_one_ip_are_blocked(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        for number in range(5):
            log_in(api, email=f"ghost{number}@college.edu")
        assert log_in(api).status_code == 429

    def test_a_block_applies_to_an_account_from_another_ip(self) -> None:
        limiter = LoginLimiter(fakeredis.FakeRedis(decode_responses=True))
        for _ in range(5):
            limiter.record_failure("1.1.1.1", EMAIL)
        assert limiter.is_blocked("2.2.2.2", EMAIL)
        assert not limiter.is_blocked("2.2.2.2", "other@college.edu")

    def test_the_window_expires_after_fifteen_minutes(self) -> None:
        client = fakeredis.FakeRedis(decode_responses=True)
        LoginLimiter(client).record_failure("1.1.1.1", EMAIL)
        assert client.ttl("login:ip:1.1.1.1") == pytest.approx(900, abs=2)

    def test_more_failures_do_not_extend_the_window(self) -> None:
        client = fakeredis.FakeRedis(decode_responses=True)
        limiter = LoginLimiter(client)
        limiter.record_failure("1.1.1.1", EMAIL)
        client.expire("login:ip:1.1.1.1", 100)
        limiter.record_failure("1.1.1.1", EMAIL)
        assert client.ttl("login:ip:1.1.1.1") <= 100

    def test_a_good_login_clears_the_account_counter(
        self, api: TestClient, make_user: Callable[..., User], limiter: LoginLimiter
    ) -> None:
        verified_user(make_user)
        for _ in range(3):
            log_in(api, password="wrong password here")
        log_in(api)
        assert not limiter.is_blocked("elsewhere", EMAIL)

    def test_a_redis_outage_leaves_login_working(self) -> None:
        class BrokenRedis:
            def mget(self, *_: object) -> None:
                raise redis.ConnectionError("down")

            def pipeline(self) -> None:
                raise redis.ConnectionError("down")

            def delete(self, *_: object) -> None:
                raise redis.ConnectionError("down")

        limiter = LoginLimiter(BrokenRedis())  # type: ignore[arg-type]
        assert limiter.is_blocked("1.1.1.1", EMAIL) is False
        limiter.record_failure("1.1.1.1", EMAIL)
        limiter.clear_account(EMAIL)


class TestLockout:
    def fail(self, db: Session, times: int) -> None:
        for _ in range(times):
            with pytest.raises(service.AuthError):
                service.login(db, EMAIL, "wrong password here")
            db.commit()

    def test_the_account_locks_after_five_failures(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = verified_user(make_user)
        self.fail(db, 5)
        db.expire_all()
        assert db.get_one(User, user.id).locked_until is not None
        with pytest.raises(service.AccountLockedError):
            service.login(db, EMAIL, PASSWORD)

    def test_the_lock_ends_after_its_time(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = verified_user(make_user)
        self.fail(db, 5)
        db.expire_all()
        db.get_one(User, user.id).locked_until = utcnow() - timedelta(seconds=1)
        db.commit()
        assert service.login(db, EMAIL, PASSWORD).access_token

    def test_lockout_holds_without_redis(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        class NeverBlocks:
            def is_blocked(self, *_: object) -> bool:
                return False

            def record_failure(self, *_: object) -> None:
                return None

            def clear_account(self, *_: object) -> None:
                return None

        verified_user(make_user)
        api.app.dependency_overrides[get_login_limiter] = NeverBlocks  # type: ignore[attr-defined]
        for _ in range(5):
            log_in(api, password="wrong password here")
        assert log_in(api).status_code == 429

    def test_a_good_login_resets_the_count(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = verified_user(make_user)
        self.fail(db, 3)
        service.login(db, EMAIL, PASSWORD)
        db.commit()
        db.expire_all()
        assert db.get_one(User, user.id).failed_login_count == 0


class TestRefresh:
    def refresh(self, client: TestClient, token: str | None = None):
        cookies = {COOKIE_NAME: token} if token else {}
        client.cookies.clear()
        client.cookies.update(cookies)
        return client.post("/api/v1/auth/refresh")

    def login_token(self, api: TestClient) -> str:
        return log_in(api).cookies[COOKIE_NAME]

    def test_rotation_returns_a_new_cookie_and_access_token(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        first = self.login_token(api)
        response = self.refresh(api, first)
        assert response.status_code == 200
        assert response.json()["access_token"]
        assert response.cookies[COOKIE_NAME] != first

    def test_refresh_tokens_are_stored_hashed(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        token = self.login_token(api)
        stored = db.scalars(select(RefreshToken.token_hash)).all()
        assert token not in stored

    def test_the_cookie_lasts_seven_days(self, db: Session, make_user: Callable[..., User]) -> None:
        verified_user(make_user)
        service.login(db, EMAIL, PASSWORD)
        db.flush()
        row = db.scalars(select(RefreshToken)).one()
        assert row.expires_at - utcnow() == pytest.approx(
            timedelta(days=7), abs=timedelta(minutes=1)
        )

    def test_reusing_a_spent_token_revokes_the_whole_family(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        stolen = self.login_token(api)
        current = self.refresh(api, stolen).cookies[COOKIE_NAME]

        assert self.refresh(api, stolen).status_code == 401
        assert self.refresh(api, current).status_code == 401

    def test_reuse_is_audited(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        stolen = self.login_token(api)
        self.refresh(api, stolen)
        self.refresh(api, stolen)
        assert db.scalar(select(AuditLog).where(AuditLog.action == "auth.refresh_reuse"))

    def test_reuse_does_not_touch_other_sessions(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        stolen = self.login_token(api)
        other_device = self.login_token(api)
        self.refresh(api, stolen)
        self.refresh(api, stolen)
        assert self.refresh(api, other_device).status_code == 200

    def test_expired_tokens_are_refused(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        verified_user(make_user)
        token = self.login_token(api)
        db.scalars(select(RefreshToken)).one().expires_at = utcnow() - timedelta(minutes=1)
        db.commit()
        assert self.refresh(api, token).status_code == 401

    def test_deactivated_users_cannot_refresh(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = verified_user(make_user)
        token = self.login_token(api)
        db.get_one(User, user.id).is_active = False
        db.commit()
        assert self.refresh(api, token).status_code == 401

    def test_no_cookie_is_refused(self, api: TestClient) -> None:
        assert self.refresh(api).status_code == 401

    def test_an_unknown_token_is_refused(self, api: TestClient) -> None:
        assert self.refresh(api, "made-up").status_code == 401

    def test_logout_ends_the_session(self, api: TestClient, make_user: Callable[..., User]) -> None:
        verified_user(make_user)
        token = self.login_token(api)
        api.cookies.clear()
        response = api.post("/api/v1/auth/logout", cookies={COOKIE_NAME: token})
        assert response.status_code == 204
        assert self.refresh(api, token).status_code == 401


class TestPasswordReset:
    def forgot(self, api: TestClient, email: str = EMAIL):
        return api.post("/api/v1/auth/forgot-password", json={"email": email})

    def reset(self, api: TestClient, token: str, password: str = NEW_PASSWORD):
        return api.post(
            "/api/v1/auth/reset-password", json={"token": token, "new_password": password}
        )

    def test_full_flow(
        self, api: TestClient, make_user: Callable[..., User], mailer: SpyMailer
    ) -> None:
        verified_user(make_user)
        assert self.forgot(api).status_code == 202
        assert self.reset(api, mailer.last_token()).status_code == 204
        assert log_in(api, password="a brand new password").status_code == 200
        assert log_in(api).status_code == 401

    def test_unknown_addresses_get_the_same_reply(self, api: TestClient, mailer: SpyMailer) -> None:
        response = self.forgot(api, "ghost@college.edu")
        assert response.status_code == 202
        assert mailer.sent == []

    def test_a_reset_link_works_only_once(
        self, api: TestClient, make_user: Callable[..., User], mailer: SpyMailer
    ) -> None:
        verified_user(make_user)
        self.forgot(api)
        token = mailer.last_token()
        assert self.reset(api, token).status_code == 204
        assert self.reset(api, token, "another new password").status_code == 401

    def test_a_reset_ends_every_session(
        self, api: TestClient, make_user: Callable[..., User], mailer: SpyMailer
    ) -> None:
        verified_user(make_user)
        refresh_token = log_in(api).cookies[COOKIE_NAME]
        self.forgot(api)
        self.reset(api, mailer.last_token())
        api.cookies.clear()
        response = api.post("/api/v1/auth/refresh", cookies={COOKIE_NAME: refresh_token})
        assert response.status_code == 401

    def test_a_reset_is_how_imported_students_get_in(
        self, api: TestClient, make_user: Callable[..., User], mailer: SpyMailer
    ) -> None:
        make_user(email=EMAIL)
        self.forgot(api)
        self.reset(api, mailer.last_token())
        assert log_in(api, password="a brand new password").status_code == 200

    def test_a_reset_clears_a_lockout(
        self, api: TestClient, db: Session, make_user: Callable[..., User], mailer: SpyMailer
    ) -> None:
        user = verified_user(make_user, locked_until=utcnow() + timedelta(minutes=10))
        self.forgot(api)
        self.reset(api, mailer.last_token())
        db.expire_all()
        assert db.get_one(User, user.id).locked_until is None

    def test_a_bad_token_is_refused(self, api: TestClient) -> None:
        assert self.reset(api, "nonsense").status_code == 401

    def test_a_token_for_an_old_password_is_refused(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = verified_user(make_user)
        assert self.reset(api, create_password_reset_token(user.id, "old-hash")).status_code == 401


class TestProfile:
    def test_update_my_profile(self, client: TestClient, make_user: Callable[..., User]) -> None:
        response = client.patch(
            "/api/v1/users/me",
            headers=auth_header(make_user()),
            json={"department": "CSE", "semester": 5, "timezone": "Europe/London"},
        )
        assert response.json()["semester"] == 5
        assert response.json()["timezone"] == "Europe/London"

    def test_a_student_cannot_change_their_role(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = client.patch(
            "/api/v1/users/me", headers=auth_header(make_user()), json={"role": "admin"}
        )
        assert response.json()["role"] == "student"

    def test_an_unknown_timezone_is_refused(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = client.patch(
            "/api/v1/users/me", headers=auth_header(make_user()), json={"timezone": "Mars/Base"}
        )
        assert response.status_code == 422


class TestAdminUsers:
    def test_search_finds_by_name_and_treats_wildcards_literally(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        make_user(full_name="Asha Rao")
        make_user(full_name="Ravi Kumar")
        headers = auth_header(make_user("admin", full_name="The Admin"))
        found = client.get("/api/v1/admin/users?q=asha", headers=headers).json()
        assert [item["full_name"] for item in found["items"]] == ["Asha Rao"]
        assert client.get("/api/v1/admin/users?q=%25", headers=headers).json()["total"] == 0

    def test_filter_by_role(self, client: TestClient, make_user: Callable[..., User]) -> None:
        make_user("faculty")
        make_user()
        headers = auth_header(make_user("admin"))
        assert client.get("/api/v1/admin/users?role=faculty", headers=headers).json()["total"] == 1

    def test_changing_a_role_is_audited(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        response = client.patch(
            f"/api/v1/admin/users/{user.id}",
            headers=auth_header(make_user("admin")),
            json={"role": "faculty"},
        )
        assert response.json()["role"] == "faculty"
        entry = db.scalar(select(AuditLog).where(AuditLog.action == "user.update"))
        assert entry is not None and entry.details == {"role": "faculty"}

    @pytest.mark.parametrize("change", [{"role": "student"}, {"is_active": False}])
    def test_admins_cannot_lock_themselves_out(
        self, client: TestClient, make_user: Callable[..., User], change: dict
    ) -> None:
        admin = make_user("admin")
        response = client.patch(
            f"/api/v1/admin/users/{admin.id}", headers=auth_header(admin), json=change
        )
        assert response.status_code == 422

    def csv(self, client: TestClient, admin: User, text: str):
        return client.post(
            "/api/v1/admin/users/import",
            headers=auth_header(admin),
            files={"file": ("students.csv", text.encode())},
        )

    def test_csv_import(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        result = self.csv(
            client,
            make_user("admin"),
            "email,full_name,roll_number,department,semester,section\n"
            "a@college.edu,Asha,R1,CSE,3,A\n"
            "b@college.edu,Bhavna,R2,ECE,5,\n",
        ).json()
        assert result == {"created": 2, "skipped": 0, "errors": []}
        student = db.scalar(select(User).where(User.email == "a@college.edu"))
        assert student is not None
        assert (student.role, student.department, student.semester) == ("student", "CSE", 3)
        assert student.password_hash is None

    def test_csv_rows_fail_one_at_a_time(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        make_user(email="taken@college.edu")
        result = self.csv(
            client,
            make_user("admin"),
            "email,full_name,semester\n"
            "good@college.edu,Good,3\n"
            "taken@college.edu,Taken,3\n"
            "bad@gmail.com,Outsider,3\n"
            "nosem@college.edu,Bad Sem,12\n"
            ",No Email,3\n",
        ).json()
        assert (result["created"], result["skipped"]) == (1, 1)
        assert len(result["errors"]) == 3
        assert "Row 4" in result["errors"][0]

    def test_a_duplicate_roll_number_is_skipped(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        make_user(roll_number="R1")
        result = self.csv(
            client, make_user("admin"), "email,full_name,roll_number\nx@college.edu,X,R1\n"
        ).json()
        assert (result["created"], result["skipped"]) == (0, 1)

    def test_a_csv_missing_columns_is_explained(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        result = self.csv(client, make_user("admin"), "name\nAsha\n").json()
        assert "Missing columns: email, full_name" in result["errors"][0]


class StubStore:
    """Records vector store calls."""

    def __init__(self) -> None:
        self.deleted: list[uuid.UUID] = []
        self.superseded: list[uuid.UUID] = []

    def delete_document(self, document_id: uuid.UUID) -> None:
        self.deleted.append(document_id)

    def mark_superseded(self, document_id: uuid.UUID) -> None:
        self.superseded.append(document_id)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> StubStore:
    cache = AnswerCache(fakeredis.FakeRedis(decode_responses=True))
    import app.api.documents as documents

    monkeypatch.setattr(documents, "get_answer_cache", lambda: cache)
    return StubStore()


@pytest.fixture
def docs_api(db: Session, store: StubStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_vector_store] = lambda: store
    return TestClient(app)


class TestDocumentPermissions:
    @pytest.mark.parametrize(
        ("role", "own_department", "uploaded_it", "allowed"),
        [
            ("admin", False, False, True),
            ("faculty", True, False, True),
            ("faculty", False, True, True),
            ("faculty", False, False, False),
            ("student", True, True, False),
        ],
    )
    def test_rule(self, role: str, own_department: bool, uploaded_it: bool, allowed: bool) -> None:
        user = User(id=uuid.uuid4(), role=role, department="CSE")
        document = Document(
            department="CSE" if own_department else "ECE",
            uploader_id=user.id if uploaded_it else uuid.uuid4(),
        )
        assert can_manage_document(user, document) is allowed

    def test_a_college_wide_document_is_not_every_faculty_members(self) -> None:
        user = User(id=uuid.uuid4(), role="faculty", department=None)
        assert not can_manage_document(user, Document(department=None, uploader_id=uuid.uuid4()))

    def test_faculty_can_delete_their_departments_document(
        self,
        docs_api: TestClient,
        db: Session,
        store: StubStore,
        make_user: Callable[..., User],
    ) -> None:
        document_id, _ = add_ready_document(db, "text", department="CSE")
        response = docs_api.delete(
            f"/api/v1/documents/{document_id}",
            headers=auth_header(make_user("faculty", department="CSE")),
        )
        assert response.status_code == 204
        assert store.deleted == [document_id]
        assert db.scalar(select(AuditLog).where(AuditLog.action == "document.delete"))

    def test_faculty_cannot_delete_another_departments_document(
        self,
        docs_api: TestClient,
        db: Session,
        store: StubStore,
        make_user: Callable[..., User],
    ) -> None:
        document_id, _ = add_ready_document(db, "text", department="CSE")
        response = docs_api.delete(
            f"/api/v1/documents/{document_id}",
            headers=auth_header(make_user("faculty", department="ECE")),
        )
        assert response.status_code == 403
        assert store.deleted == []

    def test_supersede_hides_the_vectors(
        self,
        docs_api: TestClient,
        db: Session,
        store: StubStore,
        make_user: Callable[..., User],
    ) -> None:
        old, _ = add_ready_document(db, "old")
        new, _ = add_ready_document(db, "new")
        response = docs_api.post(
            f"/api/v1/documents/{old}/supersede",
            headers=auth_header(make_user("admin")),
            json={"superseded_by_id": str(new)},
        )
        assert response.json()["superseded_by_id"] == str(new)
        assert store.superseded == [old]

    def test_a_document_cannot_replace_itself(
        self, docs_api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        document_id, _ = add_ready_document(db, "x")
        response = docs_api.post(
            f"/api/v1/documents/{document_id}/supersede",
            headers=auth_header(make_user("admin")),
            json={"superseded_by_id": str(document_id)},
        )
        assert response.status_code == 422


ANY = {"student", "faculty", "admin"}
STAFF = {"faculty", "admin"}
ADMIN = {"admin"}
V1 = "/api/v1"

# Every protected endpoint, and the roles allowed through its role check. A new endpoint
# fails `test_every_protected_route_is_in_the_matrix` until someone adds it here.
PERMISSIONS: dict[tuple[str, str], set[str]] = {
    ("GET", f"{V1}/users/me"): ANY,
    ("PATCH", f"{V1}/users/me"): ANY,
    ("GET", f"{V1}/admin/users"): ADMIN,
    ("PATCH", f"{V1}/admin/users/{{user_id}}"): ADMIN,
    ("POST", f"{V1}/admin/users/import"): ADMIN,
    ("GET", f"{V1}/admin/traces/{{trace_id}}"): ADMIN,
    ("POST", f"{V1}/documents"): STAFF,
    ("GET", f"{V1}/documents"): ANY,
    ("GET", f"{V1}/documents/{{document_id}}"): ANY,
    ("GET", f"{V1}/documents/{{document_id}}/status"): ANY,
    ("PATCH", f"{V1}/documents/{{document_id}}"): STAFF,
    ("DELETE", f"{V1}/documents/{{document_id}}"): STAFF,
    ("POST", f"{V1}/documents/{{document_id}}/reingest"): STAFF,
    ("POST", f"{V1}/documents/{{document_id}}/supersede"): STAFF,
    ("POST", f"{V1}/ask"): ANY,
    ("POST", f"{V1}/conversations"): ANY,
    ("GET", f"{V1}/conversations"): ANY,
    ("GET", f"{V1}/conversations/search"): ANY,
    ("GET", f"{V1}/conversations/export"): ANY,
    ("GET", f"{V1}/conversations/{{conversation_id}}/messages"): ANY,
    ("POST", f"{V1}/conversations/{{conversation_id}}/messages"): ANY,
    ("PATCH", f"{V1}/conversations/{{conversation_id}}"): ANY,
    ("DELETE", f"{V1}/conversations/{{conversation_id}}"): ANY,
    ("POST", f"{V1}/messages/{{message_id}}/feedback"): ANY,
    ("GET", f"{V1}/admin/feedback"): ADMIN,
    ("GET", f"{V1}/admin/analytics/overview"): ADMIN,
    ("GET", f"{V1}/admin/analytics/top-questions"): ADMIN,
    ("GET", f"{V1}/admin/analytics/unanswered"): ADMIN,
    ("GET", f"{V1}/admin/analytics/intents"): ADMIN,
    ("GET", f"{V1}/admin/analytics/top-documents"): ADMIN,
    ("GET", f"{V1}/admin/analytics/cost"): ADMIN,
    ("GET", f"{V1}/admin/analytics/knowledge-gaps"): ADMIN,
    ("GET", f"{V1}/admin/analytics/document-status"): ADMIN,
    ("POST", f"{V1}/study-plans"): ANY,
    ("GET", f"{V1}/study-plans"): ANY,
    ("GET", f"{V1}/study-plans/{{plan_id}}"): ANY,
    ("PATCH", f"{V1}/study-plans/{{plan_id}}/items/{{item_id}}"): ANY,
    ("POST", f"{V1}/study-plans/{{plan_id}}/replan"): ANY,
    ("GET", f"{V1}/study-plans/{{plan_id}}/export.ics"): ANY,
    ("DELETE", f"{V1}/study-plans/{{plan_id}}"): ANY,
    ("POST", f"{V1}/reminders"): ANY,
    ("POST", f"{V1}/reminders/parse"): ANY,
    ("GET", f"{V1}/reminders"): ANY,
    ("POST", f"{V1}/reminders/{{reminder_id}}/done"): ANY,
    ("POST", f"{V1}/reminders/{{reminder_id}}/snooze"): ANY,
    ("DELETE", f"{V1}/reminders/{{reminder_id}}"): ANY,
    ("POST", f"{V1}/assignments"): STAFF,
    ("GET", f"{V1}/assignments"): STAFF,
    ("DELETE", f"{V1}/assignments/{{task_id}}"): STAFF,
    ("GET", f"{V1}/notifications"): ANY,
    ("POST", f"{V1}/notifications/{{notification_id}}/read"): ANY,
    ("POST", f"{V1}/summaries"): ANY,
    ("POST", f"{V1}/summaries/upload"): ANY,
    ("GET", f"{V1}/summaries"): ANY,
    ("GET", f"{V1}/summaries/{{summary_id}}"): ANY,
    ("GET", f"{V1}/summaries/{{summary_id}}/export"): ANY,
}

PUBLIC = {
    ("POST", f"{V1}/auth/register"),
    ("POST", f"{V1}/auth/verify-email"),
    ("POST", f"{V1}/auth/resend-verification"),
    ("POST", f"{V1}/auth/login"),
    ("POST", f"{V1}/auth/refresh"),
    ("POST", f"{V1}/auth/logout"),
    ("POST", f"{V1}/auth/forgot-password"),
    ("POST", f"{V1}/auth/reset-password"),
    ("GET", f"{V1}/auth/providers"),
    ("GET", f"{V1}/auth/google/start"),
    ("GET", f"{V1}/auth/google/callback"),
}


def route_keys() -> set[tuple[str, str]]:
    """List every (method, path) under /api/v1, from the application's own OpenAPI document."""
    paths = create_app().openapi()["paths"]
    return {
        (method.upper(), path)
        for path, operations in paths.items()
        if path.startswith(V1)
        for method in operations
    }


def concrete(path: str) -> str:
    """Fill path parameters with random ids."""
    return re.sub(r"{\w+}", lambda _: str(uuid.uuid4()), path)


class TestPermissionMatrix:
    @pytest.fixture
    def matrix_client(self, db: Session, store: StubStore) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_llm] = lambda: FakeLLM()
        app.dependency_overrides[get_vector_store] = lambda: store
        app.dependency_overrides[get_rag_pipeline] = lambda: None
        app.dependency_overrides[get_agent_runner] = lambda: None
        app.dependency_overrides[get_embedder] = lambda: HashingEmbedder()
        return TestClient(app, raise_server_exceptions=False)

    def test_every_protected_route_is_in_the_matrix(self) -> None:
        assert len(route_keys()) > 40, "Route discovery has stopped working"
        missing = route_keys() - set(PERMISSIONS) - PUBLIC
        assert missing == set()

    def test_the_matrix_lists_no_routes_that_do_not_exist(self) -> None:
        assert (set(PERMISSIONS) | PUBLIC) - route_keys() == set()

    @pytest.mark.parametrize(("method", "path"), sorted(PERMISSIONS))
    def test_anonymous_requests_are_refused(
        self, matrix_client: TestClient, method: str, path: str
    ) -> None:
        response = matrix_client.request(method, concrete(path), json={})
        assert response.status_code == 401

    @pytest.mark.parametrize(("method", "path"), sorted(PERMISSIONS))
    def test_a_bad_token_is_refused(
        self, matrix_client: TestClient, method: str, path: str
    ) -> None:
        response = matrix_client.request(
            method, concrete(path), headers={"Authorization": "Bearer junk"}, json={}
        )
        assert response.status_code == 401

    @pytest.mark.parametrize("role", sorted(ANY))
    @pytest.mark.parametrize(("method", "path"), sorted(PERMISSIONS))
    def test_each_role_against_each_route(
        self,
        matrix_client: TestClient,
        make_user: Callable[..., User],
        method: str,
        path: str,
        role: str,
    ) -> None:
        user = make_user(role)
        response = matrix_client.request(method, concrete(path), headers=auth_header(user), json={})
        if role in PERMISSIONS[(method, path)]:
            assert response.status_code not in (401, 403), response.text
        else:
            assert response.status_code == 403

    def test_a_deactivated_user_is_refused_everywhere(
        self, matrix_client: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user("admin", is_active=False)
        response = matrix_client.get(f"{V1}/admin/users", headers=auth_header(user))
        assert response.status_code == 401


def test_production_refuses_insecure_settings() -> None:
    with pytest.raises(ValueError, match="COOKIE_SECURE"):
        Settings(environment="production", secret_key="s" * 40, cookie_secure=False)
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(
            environment="production",
            cookie_secure=True,
            secret_key="dev-only-secret-key-change-me-0123456789",
        )


def test_cors_allows_the_frontend_with_credentials(client: TestClient) -> None:
    response = client.options(
        "/api/v1/users/me",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_refuses_other_origins(client: TestClient) -> None:
    response = client.options(
        "/api/v1/users/me",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in response.headers
