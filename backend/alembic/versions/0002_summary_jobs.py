"""summary jobs

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows are finished summaries, so they are backfilled as ready.
    op.add_column(
        "summaries", sa.Column("source_name", sa.Text(), nullable=False, server_default="")
    )
    op.add_column(
        "summaries", sa.Column("status", sa.Text(), nullable=False, server_default="ready")
    )
    op.alter_column("summaries", "source_name", server_default=None)
    op.alter_column("summaries", "status", server_default=None)
    op.add_column("summaries", sa.Column("error_message", sa.Text(), nullable=True))
    op.add_column("summaries", sa.Column("duration_ms", sa.Integer(), nullable=True))
    op.alter_column("summaries", "document_id", existing_type=sa.UUID(), nullable=True)
    op.alter_column(
        "summaries",
        "content",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=True,
    )
    op.create_check_constraint(
        "ck_summaries_status_valid", "summaries", "status IN ('pending', 'ready', 'failed')"
    )


def downgrade() -> None:
    op.drop_constraint("ck_summaries_status_valid", "summaries", type_="check")
    op.execute("DELETE FROM summaries WHERE document_id IS NULL OR content IS NULL")
    op.alter_column(
        "summaries",
        "content",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
    )
    op.alter_column("summaries", "document_id", existing_type=sa.UUID(), nullable=False)
    op.drop_column("summaries", "duration_ms")
    op.drop_column("summaries", "error_message")
    op.drop_column("summaries", "status")
    op.drop_column("summaries", "source_name")
