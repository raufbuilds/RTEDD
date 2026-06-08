from datetime import date, datetime

from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, Float, Index, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from server.database import Base


class Demand(Base):
    __tablename__ = "demand"
    __table_args__ = (
        CheckConstraint("hour >= 0 AND hour <= 23", name="ck_demand_hour_range"),
        UniqueConstraint("date", "hour", name="uq_demand_date_hour"),
        Index("idx_demand_date_hour", "date", "hour"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    demand: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class ForecastCache(Base):
    __tablename__ = "forecast_cache"
    __table_args__ = (
        Index("idx_forecast_cache_status_trained", "status", "trained_at"),
    )

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    include_target_date: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="false",
    )
    signature: Mapped[object | None] = mapped_column(JSONB)
    result_json: Mapped[object | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="training")
    error: Mapped[str | None] = mapped_column(Text)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class Weather(Base):
    __tablename__ = "weather"
    __table_args__ = (
        CheckConstraint("hour >= 0 AND hour <= 23", name="ck_weather_hour_range"),
    )

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    hour: Mapped[int] = mapped_column(Integer, primary_key=True)
    temp: Mapped[float | None] = mapped_column(Float)
    humidity: Mapped[float | None] = mapped_column(Float)
    wind: Mapped[float | None] = mapped_column(Float)
    solar: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str | None] = mapped_column(Text, server_default="open-meteo")


class ProphetComponent(Base):
    __tablename__ = "prophet_components"
    __table_args__ = (
        Index("idx_prophet_components_ts_type", "timestamp", "component_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    train_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    train_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    train_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    component_type: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
