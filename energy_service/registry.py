"""Model registry: trained models are persisted once (offline), versioned, hash-checked and only served
if the operational backtest evidence says they earn it. Nothing is trained or fitted in a request path."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import joblib

from energy_service.config import ROOT, Settings
from energy_service.db import iso

FEATURE_VERSION = "operational-v1"   # bump whenever feature definitions or their legal inputs change
EVIDENCE_NAME = {"OD": "xgb_OD_train_actual", "OB": "xgb_OB"}   # registry feature set -> backtest model name


class NotEligible(Exception):
    pass


class ModelIntegrityError(Exception):
    pass


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evidence_from_results(results_dir: str, feature_set: str) -> dict:
    """Serving evidence for a feature set from results/operational_{decision,metrics}.json.
    A missing file means NO evidence (not eligible)."""
    d, m = Path(results_dir) / "operational_decision.json", Path(results_dir) / "operational_metrics.json"
    name = EVIDENCE_NAME.get(feature_set)
    if name is None or not d.exists() or not m.exists():
        return {"backtest_model": name, "eligible": False, "reason": "no operational backtest evidence"}
    decision, metrics = json.loads(d.read_text()), json.loads(m.read_text())
    v = metrics["vs_baseline"].get(name, {})
    return {"backtest_model": name, "eligible": name in decision.get("eligible_models", []), "rule": decision.get("rule"),
            "mae_diff_ci": v.get("mae_diff_ci"), "dm_p_value": (v.get("dm") or {}).get("p_value"),
            "decision_sha256": sha256_file(d), "metrics_sha256": sha256_file(m)}


def chosen_params(results_dir: str, feature_set: str) -> dict:
    meta = json.loads((Path(results_dir) / "operational_run_metadata.json").read_text())
    return meta["chosen_params"]["OD" if feature_set == "OD" else "OB"]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, timeout=10).stdout.strip()
    except Exception:
        return ""


def save_model(conn, s: Settings, model, *, kind: str, feature_set: str, columns: list[str], params: dict,
               train_start: str, train_end: str, n_rows: int, data_hash: str, evidence: dict, now: datetime) -> str:
    model_id = f"{kind}-{feature_set}-{train_end}-{data_hash[:8]}"
    d = Path(s.model_dir) / model_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "model.joblib"
    joblib.dump(model, path)
    sha = sha256_file(path)
    meta = {"model_id": model_id, "kind": kind, "feature_set": feature_set, "feature_version": FEATURE_VERSION,
            "columns": columns, "params": params, "train_start": train_start, "train_end": train_end, "n_rows": n_rows,
            "data_hash": data_hash, "model_sha256": sha, "trained_at": iso(now), "git_commit": _git_commit(),
            "python": platform.python_version(), "evidence": evidence}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    conn.execute(
        "INSERT OR REPLACE INTO models (model_id, kind, feature_set, feature_version, path, sha256, trained_at, train_start, "
        "train_end, n_rows, params_json, evidence_json, eligible, active) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, "
        "COALESCE((SELECT active FROM models WHERE model_id=?), 0))",
        (model_id, kind, feature_set, FEATURE_VERSION, str(path), sha, iso(now), train_start, train_end, n_rows,
         json.dumps(params), json.dumps(evidence), int(bool(evidence.get("eligible"))), model_id))
    conn.commit()
    return model_id


def activate(conn, model_id: str) -> None:
    row = conn.execute("SELECT feature_set, eligible FROM models WHERE model_id=?", (model_id,)).fetchone()
    if row is None:
        raise KeyError(model_id)
    if not row["eligible"]:
        raise NotEligible(f"{model_id}: operational backtest evidence does not meet the serving rule")
    conn.execute("UPDATE models SET active=0 WHERE feature_set=?", (row["feature_set"],))
    conn.execute("UPDATE models SET active=1 WHERE model_id=?", (model_id,))
    conn.commit()


def load_model(row) -> tuple[object, dict]:
    """Load a registered model after checking the file hash. Returns (model, metadata)."""
    p = Path(row["path"])
    if not p.exists():
        raise FileNotFoundError(str(p))
    if sha256_file(p) != row["sha256"]:
        raise ModelIntegrityError(f"{row['model_id']}: file hash does not match the registry")
    meta = json.loads((p.parent / "metadata.json").read_text())
    return joblib.load(p), meta
