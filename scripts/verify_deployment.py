"""Post-deployment smoke test for the forecast API. Exits non-zero if any check fails.

    python scripts/verify_deployment.py --api-url http://localhost:8000 [--allow-no-forecast] [--dashboard-url http://localhost:8501]

Checks liveness, readiness semantics, the models and forecast shapes, provenance fields, the monitoring summary, the
consistent error body, that read endpoints refuse writes, and that the admin endpoint is not open. It does not generate
anything and needs no secrets.
"""
import argparse
import sys

import httpx


def verify(client: httpx.Client, *, allow_no_forecast: bool = False, allow_not_ready: bool = False) -> list[tuple[str, bool, str]]:
    res: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        res.append((name, bool(ok), detail))

    def get(path):
        try:
            return client.get(path)
        except httpx.HTTPError as e:
            check(f"GET {path} reachable", False, str(e))
            return None

    r = get("/health")
    if r is not None:
        check("health is ok", r.status_code == 200 and r.json().get("status") == "ok", f"HTTP {r.status_code}")
        check("responses carry X-Request-ID", bool(r.headers.get("x-request-id")))
    r = get("/ready")
    if r is not None:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        ok = r.status_code == 200 and body.get("status") in ("ready", "degraded")
        check("ready (200 ready/degraded)", ok or (allow_not_ready and r.status_code == 503),
              f"HTTP {r.status_code} {body.get('status')} {body.get('reasons') or body.get('error', '')}")
        check("readiness is consistent with its status code", (r.status_code == 200) == bool(body.get("ready")))
    r = get("/models")
    if r is not None:
        j = r.json() if r.status_code == 200 else {}
        check("models lists the baseline and what serves", r.status_code == 200 and "serving_now" in j and "baseline" in j)
    r = get("/forecasts/latest")
    if r is not None:
        if r.status_code == 404 and allow_no_forecast:
            check("latest forecast (none yet, allowed)", True)
        else:
            j = r.json() if r.status_code == 200 else {}
            hrs = j.get("hours") or []
            check("latest forecast has 24 hours", r.status_code == 200 and len(hrs) == 24, f"HTTP {r.status_code}")
            check("forecast carries provenance", all(k in j for k in ("generated_at", "served_kind", "late", "inputs", "origin_ts", "version")))
            check("served_kind is model or baseline", j.get("served_kind") in ("model", "baseline"))
            check("a baseline-served forecast says why", j.get("served_kind") == "model" or bool(j.get("fallback_reason")) or not j)
    r = get("/monitoring/summary")
    if r is not None:
        j = r.json() if r.status_code == 200 else {}
        check("monitoring summary has checks", r.status_code == 200 and len(j.get("checks", [])) >= 5, f"HTTP {r.status_code}")
    r = get("/definitely-not-a-route")
    if r is not None:
        check("errors use the standard body", r.status_code == 404 and "error" in r.json())
    try:
        w = client.post("/forecasts")
        check("read endpoints refuse writes", w.status_code in (404, 405), f"HTTP {w.status_code}")
        a = client.post("/admin/forecasts/generate", json={})
        check("admin endpoint is not open without a key", a.status_code in (401, 403), f"HTTP {a.status_code}")
    except httpx.HTTPError as e:
        check("write-refusal checks reachable", False, str(e))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--dashboard-url")
    ap.add_argument("--allow-no-forecast", action="store_true")
    ap.add_argument("--allow-not-ready", action="store_true")
    a = ap.parse_args(argv)
    with httpx.Client(base_url=a.api_url, timeout=15) as c:
        res = verify(c, allow_no_forecast=a.allow_no_forecast, allow_not_ready=a.allow_not_ready)
    if a.dashboard_url:
        try:
            r = httpx.get(a.dashboard_url.rstrip("/") + "/_stcore/health", timeout=15)
            res.append(("dashboard health", r.status_code == 200, f"HTTP {r.status_code}"))
        except httpx.HTTPError as e:
            res.append(("dashboard health", False, str(e)))
    for name, ok, detail in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))
    failed = [n for n, ok, _ in res if not ok]
    print(f"{len(res) - len(failed)}/{len(res)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
