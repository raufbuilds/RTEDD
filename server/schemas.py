from datetime import date as DateType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IngestRecord(BaseModel):
    date: DateType = Field(alias="Date")
    hour: int = Field(alias="Hour", ge=0, le=23)
    demand: float = Field(alias="Ontario Demand")

    model_config = ConfigDict(populate_by_name=True)


class IngestBulkRequest(BaseModel):
    rows: list[IngestRecord]


class IngestResponse(BaseModel):
    status: str
    id: int | None = None
    reason: str | None = None


class IngestBulkResponse(BaseModel):
    status: str
    received: int
    valid: int
    saved: int
    skipped: int
    invalid: int = 0


class ForecastResponse(BaseModel):
    status: str
    target_date: DateType | None
    include_target_date: bool
    forecast: list[dict[str, Any]] = Field(default_factory=list)
    trained_at: float | None = None
    requested_at: float | None = None
    training_seconds: float | None = None
    stale: bool = False
    error: str | None = None
    message: str | None = None
    summary: str | None = None


class HealthResponse(BaseModel):
    status: str
    database: str


class ForecastCacheRow(BaseModel):
    cache_key: str
    target_date: DateType
    include_target_date: bool
    signature: Any = None
    forecast: list[dict[str, Any]] = Field(default_factory=list)
    status: str
    error: str | None = None
    trained_at: float | None = None
    requested_at: float | None = None

    @field_validator("forecast", mode="before")
    @classmethod
    def normalize_forecast(cls, value):
        return value or []
