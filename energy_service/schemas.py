"""Pydantic response models for the API."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict | list | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HourOut(BaseModel):
    hour: int = Field(ge=0, le=23, description="local hour of the delivery day (America/New_York)")
    baseline_usd_mwh: float
    model_usd_mwh: float | None = None
    served_usd_mwh: float


class EvaluationOut(BaseModel):
    mae_served: float
    mae_baseline: float
    mae_model: float | None = None
    bias_served: float
    actual_usd_mwh: list[float] | None = None


class ForecastOut(BaseModel):
    id: int
    target_date: date
    origin_date: date
    version: int
    origin_ts: datetime = Field(description="nominal forecast origin (05:00 ET on origin_date)")
    generated_at: datetime
    source: str
    served_kind: Literal["model", "baseline"]
    served_model_id: str | None
    fallback_reason: str | None
    late: bool = Field(description="generated after the day-ahead prices were public; excluded from performance claims")
    inputs: dict
    hours: list[HourOut] | None = None
    evaluation: EvaluationOut | None = None


class ForecastList(BaseModel):
    count: int
    forecasts: list[ForecastOut]


class ModelOut(BaseModel):
    model_id: str
    kind: str
    feature_set: str
    feature_version: str
    trained_at: datetime
    train_start: date | None
    train_end: date | None
    n_rows: int | None
    eligible: bool
    active: bool
    evidence: dict | None


class ModelsOut(BaseModel):
    serving_now: str = Field(description="'baseline' or the active model id")
    baseline: dict
    models: list[ModelOut]


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyOut(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    ready: bool
    reasons: list[str]
    degraded: list[str]
    schema_version: int | None = None
    latest_price_date: str | None = None
    latest_load_date: str | None = None
    active_model: str | None = None
    forecast_for_tomorrow: dict | None = None
    checked_at: str
