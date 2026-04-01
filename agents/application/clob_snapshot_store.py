"""
SQLite storage for CLOB snapshot rows (indexed by family_key and condition_id).

Prefer this over huge monolithic JSONL when you need fast family-scoped loads for backtests
and UI. Full row JSON is kept in ``payload`` so the schema matches JSONL consumers.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agents.application.clob_snapshot_backtest import (
    family_key_from_snapshot_row,
    parse_ts_utc,
)

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  ts_unix REAL NOT NULL,
  condition_id TEXT NOT NULL,
  family_key TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snap_family_ts ON snapshots(family_key, ts_unix);
CREATE INDEX IF NOT EXISTS idx_snap_cid_ts ON snapshots(condition_id, ts_unix);
CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(ts_unix);
CREATE TABLE IF NOT EXISTS clob_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


def condition_id_from_row(row: Dict[str, Any]) -> str:
    cid = (row.get("condition_id") or "").strip()
    if not cid:
        yt = (row.get("yes_token_id") or "").strip()
        nt = (row.get("no_token_id") or "").strip()
        cid = f"{yt}:{nt}" if yt and nt else ""
    return cid


def is_updown_snapshot_row(row: Dict[str, Any]) -> bool:
    """
    True for Polymarket crypto **up/down** markets (slug pattern used by hub recorders).

    Matches ``updown`` or ``up-or-down`` in ``market_slug`` or ``event_slug`` (case-insensitive).
    """
    for key in ("market_slug", "event_slug"):
        s = (str(row.get(key) or "")).lower()
        if "updown" in s or "up-or-down" in s:
            return True
    return False


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.execute(
        "INSERT OR REPLACE INTO clob_meta (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()


def row_to_tuple(row: Dict[str, Any]) -> Tuple[str, float, str, str, str]:
    ts_utc = str(row.get("ts_utc") or "")
    ts_unix = parse_ts_utc(ts_utc)
    cid = condition_id_from_row(row)
    if not cid:
        cid = "__missing_cid__"
    fk = family_key_from_snapshot_row(row)
    payload = json.dumps(row, separators=(",", ":"), ensure_ascii=False)
    return (ts_utc, ts_unix, cid, fk, payload)


def append_snapshots_batch(
    db_path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    updown_only: bool = False,
) -> int:
    """Insert snapshot dicts; returns number of rows inserted."""
    if updown_only:
        rows = [r for r in rows if is_updown_snapshot_row(r)]
    if not rows:
        return 0
    with _connect(db_path) as conn:
        init_db(conn)
        conn.executemany(
            "INSERT INTO snapshots (ts_utc, ts_unix, condition_id, family_key, payload) VALUES (?,?,?,?,?)",
            [row_to_tuple(r) for r in rows],
        )
        conn.commit()
    return len(rows)


def prune_snapshots_keep_updown_only(db_path: Path) -> Tuple[int, int]:
    """
    DELETE all rows whose JSON payload is **not** an up/down market (see ``is_updown_snapshot_row``).
    Malformed JSON rows are removed. Returns ``(rows_deleted, rows_kept)``.
    """
    if not db_path.is_file():
        return 0, 0
    deleted = 0
    kept = 0
    pending: List[int] = []

    def flush_delete(conn: sqlite3.Connection) -> None:
        nonlocal deleted, pending
        if not pending:
            return
        chunk = pending
        pending = []
        q = f"DELETE FROM snapshots WHERE id IN ({','.join('?' * len(chunk))})"
        conn.execute(q, chunk)
        deleted += len(chunk)

    with _connect(db_path) as conn:
        init_db(conn)
        cur = conn.execute("SELECT id, payload FROM snapshots")
        for sid, payload in cur:
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                pending.append(sid)
                if len(pending) >= 4000:
                    flush_delete(conn)
                continue
            if isinstance(obj, dict) and is_updown_snapshot_row(obj):
                kept += 1
            else:
                pending.append(sid)
                if len(pending) >= 4000:
                    flush_delete(conn)
        flush_delete(conn)
        conn.commit()
    return deleted, kept


def load_rows_from_sqlite(
    db_path: Path,
    *,
    family_keys: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load snapshot rows. If ``family_keys`` is set, only those ``family_key`` values (SQL IN).
    Otherwise full table ordered by ``ts_unix``.
    """
    if not db_path.is_file():
        return []
    if family_keys is not None and len(family_keys) == 0:
        return []
    out: List[Dict[str, Any]] = []
    with _connect(db_path) as conn:
        init_db(conn)
        if family_keys is None:
            cur = conn.execute(
                "SELECT payload FROM snapshots ORDER BY ts_unix ASC, id ASC"
            )
        else:
            keys = sorted(family_keys)
            placeholders = ",".join("?" * len(keys))
            cur = conn.execute(
                f"SELECT payload FROM snapshots WHERE family_key IN ({placeholders}) "
                "ORDER BY ts_unix ASC, id ASC",
                keys,
            )
        for (payload,) in cur:
            try:
                out.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return out


def list_family_keys_sqlite(db_path: Path) -> List[str]:
    if not db_path.is_file():
        return []
    with _connect(db_path) as conn:
        init_db(conn)
        cur = conn.execute(
            "SELECT DISTINCT family_key FROM snapshots ORDER BY family_key ASC"
        )
        return [r[0] for r in cur.fetchall()]


def count_rows_sqlite(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    with _connect(db_path) as conn:
        init_db(conn)
        cur = conn.execute("SELECT COUNT(*) FROM snapshots")
        return int(cur.fetchone()[0])


def migrate_jsonl_paths(
    jsonl_paths: Sequence[Path],
    db_path: Path,
    *,
    batch_size: int = 4000,
    clear_first: bool = False,
) -> Tuple[int, int]:
    """
    Stream JSONL files into ``db_path``. Returns (lines_read, rows_inserted).
    Skips malformed lines. Duplicate content from overlapping files is inserted as duplicates.
    """
    batch: List[Tuple[str, float, str, str, str]] = []
    lines_read = 0
    inserted = 0

    with _connect(db_path) as conn:
        init_db(conn)
        if clear_first:
            conn.execute("DELETE FROM snapshots")
            conn.commit()

        def flush() -> None:
            nonlocal batch, inserted
            if not batch:
                return
            conn.executemany(
                "INSERT INTO snapshots (ts_utc, ts_unix, condition_id, family_key, payload) VALUES (?,?,?,?,?)",
                batch,
            )
            conn.commit()
            inserted += len(batch)
            batch = []

        for path in jsonl_paths:
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines_read += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    batch.append(row_to_tuple(row))
                    if len(batch) >= batch_size:
                        flush()

        flush()

    return lines_read, inserted


def iter_crypto_clob_jsonl(clob_dir: Path) -> List[Path]:
    """Sorted ``crypto_clob_*.jsonl`` under ``clob_dir`` (same order as ``load_all_snapshot_files``)."""
    if not clob_dir.is_dir():
        return []
    return sorted(clob_dir.glob("crypto_clob_*.jsonl"))
