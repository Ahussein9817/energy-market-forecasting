"""Read-only forecast API. Inference never happens in a request: forecasts are produced by the scheduled job and
only retrieved here. No endpoint modifies state or triggers computation (an authenticated admin endpoint is
deliberately left to a later phase)."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import date, datetime

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from energy_service import __version__, db, status
from energy_service.config import Settings
from energy_service.schemas import (
    ErrorResponse,
    EvaluationOut,
    ForecastList,
    ForecastOut,
    HealthOut,
    HourOut,
    ModelOut,
    ModelsOut,
    ReadyOut,
)

log = logging.getLogger("energy.api")
ERR = {404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details=None):
        self.status_code, self.code, self.message, self.details = status_code, code, message, details


def _err(status_code: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "details": details}}, status_code=status_code,
                        headers=headers)


def create_app(settings: Settings | None = None, clock: Callable[[], datetime] = db.utcnow) -> FastAPI:
    s = settings or Settings.from_env()
    app = FastAPI(title="Electricity forecast service (read-only)", version=__version__,
                  description="Day-ahead NYISO N.Y.C. price forecasts, their provenance and evaluation. "
                              "A research/demonstration system, not trading advice.")

    def get_conn():
        try:
            conn = db.connect(s.db_path, read_only=True)
        except sqlite3.Error as e:
            raise ApiError(503, "db_unavailable", "database cannot be opened", {"error": str(e)}) from e
        try:
            yield conn
        finally:
            conn.close()

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            log.info(json.dumps({"event": "request", "request_id": rid, "method": request.method, "path": request.url.path,
                                 "status": status_code, "duration_ms": round((time.perf_counter() - t0) * 1000, 1)}))

    @app.exception_handler(ApiError)
    async def _api_error(_, e: ApiError):
        return _err(e.status_code, e.code, e.message, e.details)

    @app.exception_handler(StarletteHTTPException)   # also covers unknown routes and 405s
    async def _http_error(_, e: StarletteHTTPException):
        return _err(e.status_code, "http_error", str(e.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_, e: RequestValidationError):
        return _err(422, "invalid_request", "request parameters are invalid",
                    [{"loc": list(x["loc"]), "msg": x["msg"]} for x in e.errors()])

    @app.exception_handler(sqlite3.Error)
    async def _db_error(_, e: sqlite3.Error):
        return _err(503, "db_error", "database error", {"error": str(e)})

    # ---- helpers -----------------------------------------------------------------------------
    def to_forecast(conn, row, with_hours: bool) -> ForecastOut:
        hours = None
        if with_hours:
            hours = [HourOut(hour=h["hour"], baseline_usd_mwh=h["baseline"], model_usd_mwh=h["model"], served_usd_mwh=h["served"])
                     for h in conn.execute("SELECT * FROM forecast_hours WHERE forecast_id=? ORDER BY hour", (row["id"],))]
        ev = conn.execute("SELECT * FROM evaluations WHERE forecast_id=?", (row["id"],)).fetchone()
        evaluation = None
        if ev is not None:
            actual = [r[0] for r in conn.execute("SELECT actual FROM evaluation_hours WHERE forecast_id=? ORDER BY hour", (row["id"],))]
            evaluation = EvaluationOut(mae_served=ev["mae_served"], mae_baseline=ev["mae_baseline"], mae_model=ev["mae_model"],
                                       bias_served=ev["bias_served"], actual_usd_mwh=actual if with_hours else None)
        return ForecastOut(id=row["id"], target_date=row["target_date"], origin_date=row["origin_date"], version=row["version"],
                           origin_ts=row["origin_ts"], generated_at=row["generated_at"], source=row["source"],
                           served_kind=row["served_kind"], served_model_id=row["served_model_id"],
                           fallback_reason=row["fallback_reason"], late=bool(row["late"]), inputs=json.loads(row["input_json"]),
                           hours=hours, evaluation=evaluation)

    # ---- endpoints ---------------------------------------------------------------------------
    @app.get("/health", response_model=HealthOut, summary="Liveness: the process is up")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/ready", response_model=ReadyOut, responses={503: {"model": ReadyOut}},
             summary="Readiness: can a trustworthy baseline be served? 503 when not")
    def ready(conn=Depends(get_conn)):
        r = status.readiness(conn, s, clock())
        return JSONResponse(r, status_code=200 if r["ready"] else 503)

    @app.get("/models", response_model=ModelsOut, summary="Registered models, which one serves, and the evidence")
    def models(conn=Depends(get_conn)):
        rows = conn.execute("SELECT * FROM models ORDER BY trained_at DESC").fetchall()
        out = [ModelOut(model_id=r["model_id"], kind=r["kind"], feature_set=r["feature_set"], feature_version=r["feature_version"],
                        trained_at=r["trained_at"], train_start=r["train_start"], train_end=r["train_end"], n_rows=r["n_rows"],
                        eligible=bool(r["eligible"]), active=bool(r["active"]),
                        evidence=json.loads(r["evidence_json"]) if r["evidence_json"] else None) for r in rows]
        active = next((m.model_id for m in out if m.active), None)
        return ModelsOut(serving_now=active or "baseline",
                         baseline={"name": "same_hour_of_day_D", "description": "price of day D repeated for D+1, always available"},
                         models=out)

    @app.get("/forecasts/latest", response_model=ForecastOut, responses=ERR, summary="Most recent forecast, with hours")
    def latest(conn=Depends(get_conn)):
        row = conn.execute("SELECT * FROM forecasts ORDER BY target_date DESC, version DESC LIMIT 1").fetchone()
        if row is None:
            raise ApiError(404, "no_forecast", "no forecast has been generated yet")
        return to_forecast(conn, row, True)

    @app.get("/forecasts", response_model=ForecastList, responses=ERR, summary="Forecast history (latest version per day)")
    def forecasts(target_date: date | None = None, date_from: date | None = None, date_to: date | None = None,
                  limit: int = Query(30, ge=1, le=366), include_hours: bool = False, conn=Depends(get_conn)):
        if target_date and (date_from or date_to):
            raise ApiError(422, "invalid_request", "use either target_date or date_from/date_to, not both")
        if date_from and date_to and date_from > date_to:
            raise ApiError(422, "invalid_request", "date_from is after date_to")
        where, args = ["f.version = (SELECT MAX(version) FROM forecasts WHERE target_date=f.target_date)"], []
        if target_date:
            where.append("f.target_date=?")
            args.append(str(target_date))
        if date_from:
            where.append("f.target_date>=?")
            args.append(str(date_from))
        if date_to:
            where.append("f.target_date<=?")
            args.append(str(date_to))
        rows = conn.execute(f"SELECT f.* FROM forecasts f WHERE {' AND '.join(where)} ORDER BY f.target_date DESC LIMIT ?",
                            (*args, limit)).fetchall()
        if target_date and not rows:
            raise ApiError(404, "no_forecast", f"no forecast for {target_date}")
        return ForecastList(count=len(rows), forecasts=[to_forecast(conn, r, include_hours) for r in rows])

    return app
