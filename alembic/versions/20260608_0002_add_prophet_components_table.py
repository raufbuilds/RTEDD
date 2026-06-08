"""add prophet_components table

Revision ID: 20260608_0002
Revises: 20260608_0001
Create Date: 2026-06-08
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op


revision: str = "20260608_0002"
down_revision: str | None = "20260608_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prophet_components",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("fitted_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("train_start_date", sa.Date(), nullable=False),
        sa.Column("train_end_date", sa.Date(), nullable=False),
        sa.Column("train_row_count", sa.Integer(), nullable=False),
        sa.Column("component_type", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.REAL(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_prophet_components_ts_type",
        "prophet_components",
        ["timestamp", "component_type"],
    )


def downgrade() -> None:
    op.drop_index("idx_prophet_components_ts_type", table_name="prophet_components")
    op.drop_table("prophet_components")
