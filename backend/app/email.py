"""Sending email, behind an interface."""

import hashlib
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid, parseaddr
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)


class EmailSender(Protocol):
    """Sends one plain-text email."""

    def send(self, to: str, subject: str, body: str, message_id: str | None = None) -> None:
        """Send a message, raising if it fails.

        `message_id` is a stable key for this message. Sending again with the same key gives
        the same Message-ID header, so mail clients that deduplicate show one copy.
        """
        ...


class SmtpEmailSender:
    """Sends through an SMTP server over STARTTLS."""

    def __init__(self, host: str, port: int, username: str, password: str, sender: str) -> None:
        """Keep the connection details."""
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._sender = sender

    def _message_id(self, key: str | None) -> str:
        """Build the Message-ID header, derived from the key when there is one."""
        domain = parseaddr(self._sender)[1].rpartition("@")[2] or "campusmind.local"
        if key is None:
            return make_msgid(domain=domain)
        return f"<{hashlib.sha256(key.encode()).hexdigest()[:32]}@{domain}>"

    def send(self, to: str, subject: str, body: str, message_id: str | None = None) -> None:
        """Open a connection and send."""
        message = EmailMessage()
        message["Message-ID"] = self._message_id(message_id)
        message["From"] = self._sender
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        with smtplib.SMTP(self._host, self._port, timeout=20) as server:
            server.starttls()
            if self._username:
                server.login(self._username, self._password)
            server.send_message(message)


class LoggingEmailSender:
    """Logs instead of sending, for development when SMTP is not configured."""

    def send(self, to: str, subject: str, body: str, message_id: str | None = None) -> None:
        """Record that a message would have been sent."""
        log.info("email_not_sent_smtp_unconfigured", to=to, subject=subject, chars=len(body))


@lru_cache
def get_email_sender() -> EmailSender:
    """Return SMTP if it is configured, otherwise the logging sender."""
    settings = get_settings()
    if not settings.smtp_host:
        return LoggingEmailSender()
    return SmtpEmailSender(
        settings.smtp_host,
        settings.smtp_port,
        settings.smtp_username,
        settings.smtp_password,
        settings.smtp_from,
    )
