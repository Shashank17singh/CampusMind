"""Sending email, behind an interface."""

import smtplib
from email.message import EmailMessage
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)


class EmailSender(Protocol):
    """Sends one plain-text email."""

    def send(self, to: str, subject: str, body: str) -> None:
        """Send a message, raising if it fails."""
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

    def send(self, to: str, subject: str, body: str) -> None:
        """Open a connection and send."""
        message = EmailMessage()
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

    def send(self, to: str, subject: str, body: str) -> None:
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
