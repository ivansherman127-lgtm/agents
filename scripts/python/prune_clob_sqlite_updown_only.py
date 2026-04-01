#!/usr/bin/env python3
"""
Remove from the CLOB SQLite store every snapshot row that is **not** an up/down market
(``updown`` / ``up-or-down`` in ``market_slug`` or ``event_slug``).

Does not change JSONL files. Optional ``--vacuum`` reclaims disk space (can be slow).

Example:

  python scripts/python/prune_clob_sqlite_updown_only.py \\
    --db data/clob_snapshots/clob_snapshots.db --vacuum
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.application.clob_snapshot_store import prune_snapshots_keep_updown_only  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Keep only up/down markets in clob_snapshots SQLite.")
    p.add_argument(
        "--db",
        type=Path,
        default=REPO_ROOT / "data" / "clob_snapshots" / "clob_snapshots.db",
        help="Path to SQLite file",
    )
    p.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after delete (reclaim space; may take a while)",
    )
    args = p.parse_args()
    db = args.db.expanduser()
    if not db.is_file():
        print(f"Not found: {db}", file=sys.stderr)
        sys.exit(1)
    deleted, kept = prune_snapshots_keep_updown_only(db)
    print(f"Deleted {deleted:,} rows; kept {kept:,} up/down rows.", flush=True)
    if args.vacuum:
        import sqlite3

        conn = sqlite3.connect(str(db), timeout=120.0)
        conn.execute("VACUUM")
        conn.close()
        print("VACUUM done.", flush=True)


if __name__ == "__main__":
    main()
