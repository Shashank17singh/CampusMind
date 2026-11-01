"""Object-level permission checks."""

from app.models import Document, User


def can_manage_document(user: User, document: Document) -> bool:
    """Admins manage everything. Faculty manage their uploads and their department's."""
    if user.role == "admin":
        return True
    if user.role != "faculty":
        return False
    if document.uploader_id == user.id:
        return True
    return document.department is not None and document.department == user.department
