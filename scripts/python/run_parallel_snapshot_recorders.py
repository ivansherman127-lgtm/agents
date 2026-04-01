#!/usr/bin/env python3
"""
Run two CLOB snapshot streams in parallel:
1) 15M markets
2) Hourly up/down-only markets

Designed for high-frequency local recording with independent worker pools.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.application.polymarket_crypto_pages import CryptoMarketTarget, all_hub_targets
from scripts.python.record_crypto_clob_snapshots import run_sweep


def _is_updown_slug(slug: str) -> bool:
    s = (slug or "").lower()
    return ("updown" in s) or ("up-or-down" in s)


def _discover_15m_targets() -> List[CryptoMarketTarget]:
    return all_hub_targets(["15M"])


def _discover_hourly_updown_targets() -> List[CryptoMarketTarget]:
    return [t for t in all_hub_targets(["hourly"]) if _is_updown_slug(t.market_slug)]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parallel 15M + hourly up/down CLOB recorder."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "clob_snapshots",
        help="Output directory for daily JSONL files.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between parallel sweep cycles.",
    )
    p.add_argument(
        "--workers-15m",
        type=int,
        default=8,
        help="Worker threads for 15M stream.",
    )
    p.add_argument(
        "--workers-1h",
        type=int,
        default=8,
        help="Worker threads for hourly up/down stream.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout per request (seconds).",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Max bids/asks levels per side (default: full).",
    )
    p.add_argument(
        "--with-midpoint",
        action="store_true",
        help="Also fetch midpoint for YES/NO tokens.",
    )
    p.add_argument("--once", action="store_true", help="Run one parallel sweep and exit.")
    p.add_argument(
        "--sqlite-db",
        type=Path,
        default=None,
        help="Also append sweeps to this SQLite file (same as record_crypto_clob_snapshots.py).",
    )
    p.add_argument(
        "--sqlite-all-markets",
        action="store_true",
        help="With --sqlite-db: insert every market into SQLite (default: only up/down slugs).",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_n = 0
    sqlite_note = ""
    if args.sqlite_db:
        sqlite_note = (
            f" sqlite={args.sqlite_db} (all markets)"
            if args.sqlite_all_markets
            else f" sqlite={args.sqlite_db} (up/down only)"
        )
    print(
        f"Parallel recorder -> out={args.out_dir} interval={args.interval}s "
        f"workers(15m/1h)={args.workers_15m}/{args.workers_1h}{sqlite_note}",
        flush=True,
    )

    while True:
        sweep_n += 1
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = args.out_dir / f"crypto_clob_{day}.jsonl"
        t0 = time.monotonic()

        try:
            t15 = _discover_15m_targets()
        except Exception as e:
            print(f"[sweep {sweep_n}] 15M discovery failed: {e}", flush=True)
            t15 = []
        try:
            t1h = _discover_hourly_updown_targets()
        except Exception as e:
            print(f"[sweep {sweep_n}] 1H up/down discovery failed: {e}", flush=True)
            t1h = []

        def run_15m():
            if not t15:
                return (0, 0, 0.0)
            s0 = time.monotonic()
            ok, err = run_sweep(
                t15,
                out_path,
                args.depth,
                args.with_midpoint,
                args.timeout,
                max(1, args.workers_15m),
                sqlite_db=args.sqlite_db,
                sqlite_updown_only=not args.sqlite_all_markets,
            )
            return (ok, err, time.monotonic() - s0)

        def run_1h():
            if not t1h:
                return (0, 0, 0.0)
            s0 = time.monotonic()
            ok, err = run_sweep(
                t1h,
                out_path,
                args.depth,
                args.with_midpoint,
                args.timeout,
                max(1, args.workers_1h),
                sqlite_db=args.sqlite_db,
                sqlite_updown_only=not args.sqlite_all_markets,
            )
            return (ok, err, time.monotonic() - s0)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f15 = ex.submit(run_15m)
            f1h = ex.submit(run_1h)
            ok15, err15, dt15 = f15.result()
            ok1h, err1h, dt1h = f1h.result()

        dt = time.monotonic() - t0
        print(
            f"[sweep {sweep_n}] 15M {ok15}/{len(t15)} rows (err={err15}, {dt15:.3f}s) | "
            f"1H up/down {ok1h}/{len(t1h)} rows (err={err1h}, {dt1h:.3f}s) | "
            f"total {ok15 + ok1h} rows in {dt:.3f}s -> {out_path.name}",
            flush=True,
        )

        if args.once:
            break
        sleep_s = max(0.05, args.interval - dt)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()

