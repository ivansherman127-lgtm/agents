#!/usr/bin/env python3
"""
Poll Polymarket crypto hub pages (15M / hourly / 4h), discover all CLOB markets, and append
order-book snapshots to daily JSONL files under data/clob_snapshots/.

Requires only the public CLOB read API (no wallet).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.application.polymarket_crypto_pages import (  # noqa: E402
    DEFAULT_UA,
    CryptoMarketTarget,
    all_hub_targets,
)


def _fetch_json(url: str, timeout_s: float) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def fetch_book(token_id: str, timeout_s: float) -> Optional[Dict[str, Any]]:
    q = urllib.parse.urlencode({"token_id": token_id})
    return _fetch_json(f"https://clob.polymarket.com/book?{q}", timeout_s)


def fetch_midpoint(token_id: str, timeout_s: float) -> Optional[str]:
    q = urllib.parse.urlencode({"token_id": token_id})
    data = _fetch_json(f"https://clob.polymarket.com/midpoint?{q}", timeout_s)
    if not data:
        return None
    mid = data.get("mid")
    return str(mid) if mid is not None else None


def truncate_book(book: Optional[Dict[str, Any]], depth: Optional[int]) -> Optional[Dict[str, Any]]:
    if book is None or depth is None:
        return book
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not isinstance(bids, list):
        bids = []
    if not isinstance(asks, list):
        asks = []
    out = dict(book)
    out["bids"] = bids[:depth]
    out["asks"] = asks[:depth]
    return out


def _one_market_snapshot(
    t: CryptoMarketTarget,
    depth: Optional[int],
    with_midpoint: bool,
    timeout_s: float,
) -> Tuple[CryptoMarketTarget, Dict[str, Any]]:
    yes_b = fetch_book(t.yes_token_id, timeout_s)
    no_b = fetch_book(t.no_token_id, timeout_s)
    row: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bucket": t.bucket,
        "page_key": t.page_key,
        "event_slug": t.event_slug,
        "event_title": t.event_title,
        "market_slug": t.market_slug,
        "condition_id": t.condition_id,
        "yes_token_id": t.yes_token_id,
        "no_token_id": t.no_token_id,
        "yes_book": truncate_book(yes_b, depth),
        "no_book": truncate_book(no_b, depth),
    }
    if with_midpoint:
        row["yes_mid"] = fetch_midpoint(t.yes_token_id, timeout_s)
        row["no_mid"] = fetch_midpoint(t.no_token_id, timeout_s)
    return t, row


def run_sweep(
    targets: List[CryptoMarketTarget],
    out_path: Path,
    depth: Optional[int],
    with_midpoint: bool,
    timeout_s: float,
    workers: int,
) -> Tuple[int, int]:
    ok = 0
    err = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(
        max_workers=max(1, workers)
    ) as ex:
        futs = {
            ex.submit(_one_market_snapshot, t, depth, with_midpoint, timeout_s): t
            for t in targets
        }
        for fut in as_completed(futs):
            try:
                _, row = fut.result()
                f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            except Exception:
                err += 1
    return ok, err


def main() -> None:
    p = argparse.ArgumentParser(description="Record CLOB book snapshots for Polymarket crypto hubs.")
    p.add_argument(
        "--pages",
        default="15M,hourly,4hour",
        help="Comma-separated page keys: 15M,hourly,4hour (default: all)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "clob_snapshots",
        help="Directory for daily JSONL files",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between sweeps (lower is denser data for intrabar backtest realism)",
    )
    p.add_argument("--once", action="store_true", help="Single sweep then exit")
    p.add_argument("--depth", type=int, default=None, help="Max bids/asks levels per side (default: full)")
    p.add_argument("--workers", type=int, default=24, help="Parallel markets per sweep")
    p.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout per request (seconds)")
    p.add_argument(
        "--with-midpoint",
        action="store_true",
        help="Also fetch /midpoint for YES and NO (extra 2 HTTP calls per market)",
    )
    args = p.parse_args()
    page_keys = [x.strip() for x in args.pages.split(",") if x.strip()]

    print(f"Recording to {args.out_dir} every {args.interval}s; pages={page_keys}", flush=True)
    sweep_n = 0
    while True:
        sweep_n += 1
        try:
            targets = all_hub_targets(page_keys)
        except Exception as e:
            print(f"[sweep {sweep_n}] discovery failed: {e}", flush=True)
            targets = []
        if not targets:
            print(f"[sweep {sweep_n}] no markets (empty hub or fetch error); retrying…", flush=True)
        else:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            out_path = args.out_dir / f"crypto_clob_{day}.jsonl"
            t0 = time.monotonic()
            ok, err = run_sweep(
                targets,
                out_path,
                args.depth,
                args.with_midpoint,
                args.timeout,
                args.workers,
            )
            dt = time.monotonic() - t0
            print(
                f"[sweep {sweep_n}] wrote {ok} rows to {out_path.name} "
                f"({len(targets)} markets, {dt:.1f}s, errors={err})",
                flush=True,
            )
        if args.once:
            break
        time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    main()
