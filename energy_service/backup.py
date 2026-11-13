"""Consistent SQLite backup and a guarded restore."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

TABLES = ("prices_dam", "load_actual", "forecasts", "forecast_hours", "evaluations", "models", "raw_files", "events")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _counts(path: str) -> dict:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        c.close()


def _integrity(path: str) -> str:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return c.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        c.close()


def backup(db_path: str, out_path: str) -> dict:
    """Online backup (safe while the service runs), verified, with a checksum sidecar `<out>.json`."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    dst = sqlite3.connect(str(out))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    ok = _integrity(str(out))
    if ok != "ok":
        out.unlink()
        raise RuntimeError(f"backup failed integrity check: {ok}")
    meta = {"source": str(db_path), "sha256": _sha(out), "bytes": out.stat().st_size, "counts": _counts(str(out))}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
    return meta


def restore(backup_path: str, db_path: str, force: bool = False) -> dict:
    """Verify the backup against its checksum and integrity, then replace the database atomically.
    Refuses to overwrite an existing database unless force=True (the old file is kept as `<db>.before-restore`)."""
    b = Path(backup_path)
    side = Path(str(b) + ".json")
    if side.exists():
        meta = json.loads(side.read_text())
        if _sha(b) != meta["sha256"]:
            raise ValueError("backup file does not match its recorded checksum")
    if _integrity(str(b)) != "ok":
        raise ValueError("backup failed integrity check")
    dest = Path(db_path)
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} exists; pass force to replace it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".restoring")
    shutil.copyfile(b, tmp)
    if dest.exists():
        shutil.copyfile(dest, str(dest) + ".before-restore")
    for ext in ("-wal", "-shm"):
        Path(str(dest) + ext).unlink(missing_ok=True)
    os.replace(tmp, dest)
    return {"restored_to": str(dest), "counts": _counts(str(dest))}
