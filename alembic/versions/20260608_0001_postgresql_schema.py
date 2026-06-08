"""create postgresql schema

Revision ID: 20260608_0001
Revises:
Create Date: 2026-06-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision: str = "20260608_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "demand",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("hour", sa.Integer(), nullable=False),
        sa.Column("demand", sa.REAL(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("hour >= 0 AND hour <= 23", name="ck_demand_hour_range"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date", "hour", name="uq_demand_date_hour"),
    )
    op.create_index("idx_demand_date_hour", "demand", ["date", "hour"])

    op.create_table(
        "forecast_cache",
        sa.Column("cache_key", sa.Text(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("include_target_date", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("signature", postgresql.JSONB(), nullable=True),
        sa.Column("result_json", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.Text(), server_default=sa.text("'training'"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("requested_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("cache_key"),
    )
    op.create_index(
        "idx_forecast_cache_status_trained",
        "forecast_cache",
        ["status", sa.text("trained_at DESC")],
    )

    op.create_table(
        "weather",
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("hour", sa.Integer(), nullable=False),
        sa.Column("temp", sa.REAL(), nullable=True),
        sa.Column("humidity", sa.REAL(), nullable=True),
        sa.Column("wind", sa.REAL(), nullable=True),
        sa.Column("solar", sa.REAL(), nullable=True),
        sa.Column("data_source", sa.Text(), server_default=sa.text("'open-meteo'")),
        sa.CheckConstraint("hour >= 0 AND hour <= 23", name="ck_weather_hour_range"),
        sa.PrimaryKeyConstraint("date", "hour"),
    )


def downgrade() -> None:
    op.drop_table("weather")
    op.drop_index("idx_forecast_cache_status_trained", table_name="forecast_cache")
    op.drop_table("forecast_cache")
    op.drop_index("idx_demand_date_hour", table_name="demand")
    op.drop_table("demand")
