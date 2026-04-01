#!/usr/bin/env python3
"""
Split monolithic CLOB snapshot JSONL files into **one file per market family** for faster,
targeted loads (e.g. hedge UI, single-product backtests).

  PYTHONPATH=. python scripts/python/split_clob_snapshots_by_family.py \\
    --input-dir data/clob_snapshots --out-dir data/clob_snapshots_by_family

``--input-dir`` / ``--input-file`` / ``--out-dir`` may be relative; they are resolved from the
**repository root** (next to ``agents/``), not the shell's current directory.

Input: ``crypto_clob_*.jsonl`` (same as load_all_snapshot_files).
Output: ``<bucket>_<slug-core>.jsonl`` per :func:`family_group_key` / :func:`family_key_to_split_filename`.

Rows are written in snapshot time order within each family file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _repo_root() -> Path:
    """Repository root (folder containing ``agents/application/``)."""
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "agents" / "application" / "clob_snapshot_backtest.py").is_file():
            return p
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_path(p: Path) -> Path:
    """Relative paths are resolved against the repo root (so cwd does not matter)."""
    p = p.expanduser()
    if not p.is_absolute():
        return (REPO_ROOT / p).resolve()
    return p.resolve()


from agents.application.clob_snapshot_backtest import (
    family_key_to_split_filename,
    index_family_chains,
    index_recorded_markets,
    load_all_snapshot_files,
    load_jsonl_snapshots,
    rows_for_condition,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Split CLOB JSONL into one file per market family.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "clob_snapshots",
        help="Folder containing crypto_clob_*.jsonl",
    )
    p.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single file instead of all crypto_clob_*.jsonl in input-dir",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "clob_snapshots_by_family",
        help="Output directory (created if missing)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print counts only")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print resolved paths and glob matches",
    )
    args = p.parse_args()

    input_dir = _resolve_path(args.input_dir)
    out_dir = _resolve_path(args.out_dir)

    if args.input_file:
        input_file = _resolve_path(args.input_file)
        if not input_file.is_file():
            print(f"Input file not found: {input_file}", file=sys.stderr)
            sys.exit(1)
        rows_all = load_jsonl_snapshots(input_file)
        if args.verbose:
            print(f"Loaded from file: {input_file} ({len(rows_all)} rows)", flush=True)
    else:
        if args.verbose:
            globs = sorted(input_dir.glob("crypto_clob_*.jsonl")) if input_dir.is_dir() else []
            print(f"Input dir (resolved): {input_dir}  exists={input_dir.is_dir()}", flush=True)
            print(f"crypto_clob_*.jsonl matches: {len(globs)} file(s)", flush=True)
            if globs:
                for g in globs[:8]:
                    print(f"  {g.name}", flush=True)
                if len(globs) > 8:
                    print(f"  ... +{len(globs) - 8} more", flush=True)
        rows_all = load_all_snapshot_files(input_dir)
        if args.verbose:
            print(f"Total rows merged: {len(rows_all)}", flush=True)

    if not rows_all:
        print(
            "No rows loaded. Use an existing directory with crypto_clob_*.jsonl, or pass "
            "--input-file path/to/file.jsonl. Relative paths are resolved from the repo root:\n"
            f"  {REPO_ROOT}",
            file=sys.stderr,
        )
        sys.exit(1)

    idx = index_recorded_markets(rows_all)
    chains = index_family_chains(idx)
    if args.verbose:
        print(f"Market windows (condition groups): {len(idx)}  families: {len(chains)}", flush=True)

    if not chains:
        print(
            "No market families found (no condition_id / token pairs in rows). Check snapshot schema.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    for fk, cids in sorted(chains.items()):
        fname = family_key_to_split_filename(fk)
        out_path = out_dir / fname
        lines: list[str] = []
        for cid in cids:
            for row in rows_for_condition(rows_all, cid):
                lines.append(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
        if args.dry_run:
            print(f"{fk} -> {fname} ({len(lines)} rows)")
        else:
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_lines += len(lines)

    print(
        f"Families: {len(chains)}  total rows written: {total_lines}  -> {out_dir}",
        flush=True,
    )
    if args.dry_run:
        print("(dry-run: no files written)", flush=True)


if __name__ == "__main__":
    main()
