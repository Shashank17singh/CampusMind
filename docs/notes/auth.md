# Authentication and access control

Code: `backend/app/auth/`, `backend/app/api/auth.py`, `backend/app/api/users.py`.

## How to run

Set `ALLOWED_EMAIL_DOMAIN`, `SECRET_KEY`, `FRONTEND_URL` and `CORS_ORIGINS`. Set `COOKIE_SECURE=true` in production, where startup refuses to run without it.

To create the first admin, register normally, then promote the account in the database. The seed script creates demo accounts for each role.

## How to test

```bash
pytest tests/test_auth.py
```

`TestPermissionMatrix` reads the route list from the app's own OpenAPI document and runs every role against every route. A new endpoint fails the suite until someone records who may use it. Redis is faked, and SMTP is stubbed.

## Key decisions

**1. A short JWT plus a rotating opaque refresh token.** A JWT cannot be revoked before it expires, so it lives for only 15 minutes. The refresh token is a random string stored as a hash, so it can be revoked, and a database leak does not reveal usable tokens. It sits in an `httpOnly`, `SameSite=Strict` cookie, which page scripts cannot read, so an XSS bug cannot steal it. *Rejected:* a long-lived JWT in `localStorage`, which any injected script can read and which cannot be revoked.

**2. Reuse of a refresh token revokes the whole family.** Each token works once. If a spent token comes back, then either the real user or a thief has a stale copy, and the server cannot tell which. Revoking every token descended from that login stops the thief, and the only cost is that the real user logs in again. A row lock means two simultaneous refreshes cannot both spend one token. Other devices have their own family and stay signed in.

**3. Two layers of login limits.** The Redis limiter counts per IP and per account, and it stops one attacker trying many accounts. The lockout lives in PostgreSQL, so brute force stays bounded even with Redis down. If Redis fails, login keeps working. *Rejected:* failing closed, where a Redis outage would lock out every user.

## More design points

- **Why Argon2?** It is memory-hard, which makes GPU cracking expensive. Unlike SHA-256, it is designed to be slow.
- **No account enumeration.** Registration and forgot-password reply identically whether or not the address exists. Login gives the same message for a wrong password and an unknown user, and it runs a dummy hash so timing does not reveal the difference.
- **Reset tokens are single-use without a table.** They carry a fingerprint of the current password hash, so changing the password invalidates them automatically.
- **Object-level checks.** `require_role` is too coarse, because a faculty member may only manage their own department's documents. That rule is in `can_manage_document`.
- **Why 404 and not 403 on other users' data?** A 403 confirms that the item exists.
- **Deactivation is immediate.** `get_current_user` loads the user on every request and checks `is_active`, so a valid token stops working as soon as the account is disabled. A role change also applies at once, because role checks read the database, not the token.
- **Imported students have no password.** They use forgot-password, which also proves they own the address.

## Known limits

- Two browser tabs refreshing at the same moment look like token reuse, and sign the user out. The frontend must allow only one refresh at a time.
- The IP limit is meaningless behind a proxy unless the real address is forwarded, which the production setup does with `--proxy-headers`.
- Google sign-in has only been tested against a stand-in for Google, because real use needs a Google Cloud client ID.
- **Google sign-in checks.** PKCE, a state and a nonce, carried in a signed cookie that lasts 10 minutes. The ID token comes straight from Google's token endpoint over TLS, so its signature is not checked again (OpenID Connect Core 3.1.3.7), but its issuer, audience, expiry, nonce, `email_verified` and the college domain all are. The state cookie is `SameSite=Lax`, because a `Strict` cookie is not sent with Google's cross-site redirect back.
