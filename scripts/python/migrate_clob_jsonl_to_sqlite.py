#!/usr/bin/env python3
"""
Import existing ``crypto_clob_*.jsonl`` snapshots into a SQLite database with indexed
``family_key`` and ``condition_id`` for faster backtests and tooling.

Example:

  python scripts/python/migrate_clob_jsonl_to_sqlite.py \\
    --clob-dir data/clob_snapshots \\
    --db data/clob_snapshots/clob_snapshots.db

Use ``--clear`` to wipe the ``snapshots`` table before import (avoid duplicates on re-run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.application.clob_snapshot_store import (  # noqa: E402
    iter_crypto_clob_jsonl,
    migrate_jsonl_paths,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Migrate CLOB JSONL snapshots to SQLite.")
    p.add_argument(
        "--clob-dir",
        type=Path,
        default=REPO_ROOT / "data" / "clob_snapshots",
        help="Directory containing crypto_clob_*.jsonl",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=REPO_ROOT / "data" / "clob_snapshots" / "clob_snapshots.db",
        help="Output SQLite path",
    )
    p.add_argument(
        "--files",
        type=str,
        default="",
        help="Optional comma-separated explicit JSONL paths (overrides --clob-dir glob)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4000,
        help="Rows per INSERT batch",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="DELETE all rows from snapshots before import",
    )
    args = p.parse_args()

    if args.files.strip():
        paths = [Path(x.strip()) for x in args.files.split(",") if x.strip()]
    else:
        paths = iter_crypto_clob_jsonl(args.clob_dir.expanduser())

    if not paths:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Importing {len(paths)} file(s) -> {args.db}", flush=True)
    lines, ins = migrate_jsonl_paths(
        paths,
        args.db.expanduser(),
        batch_size=max(100, args.batch_size),
        clear_first=args.clear,
    )
    print(f"Done: lines_read={lines:,} rows_inserted={ins:,}", flush=True)


if __name__ == "__main__":
    main()
