"""
Backtest a one-shot dual-limit strategy using recorded CLOB order-book snapshots.

Fill rules (conservative taker-style at touch):
- BUY YES/NO at limit L when best ask <= L; fill at the quoted best ask price.
- SELL YES/NO at limit S when best bid >= S; fill at the quoted best bid price.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def best_bid_ask(book: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Best bid (highest) and best ask (lowest) from a CLOB /book payload."""
    if not book or not isinstance(book, dict):
        return None, None
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    if isinstance(bids, list):
        for x in bids:
            if not isinstance(x, dict):
                continue
            try:
                p = float(x.get("price"))
            except (TypeError, ValueError):
                continue
            best_bid = p if best_bid is None else max(best_bid, p)
    if isinstance(asks, list):
        for x in asks:
            if not isinstance(x, dict):
                continue
            try:
                p = float(x.get("price"))
            except (TypeError, ValueError):
                continue
            best_ask = p if best_ask is None else min(best_ask, p)
    return best_bid, best_ask


def parse_ts_utc(ts: str) -> float:
    s = (ts or "").strip().replace("Z", "+00:00")
    if not s:
        return 0.0
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


@dataclass(frozen=True)
class RecordedMarketInfo:
    condition_id: str
    market_slug: str
    event_slug: str
    event_title: str
    bucket: str
    page_key: str
    n_rows: int
    t_first_iso: str
    t_last_iso: str


def load_jsonl_snapshots(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_all_snapshot_files(clob_dir: Path) -> List[Dict[str, Any]]:
    """All rows from ``crypto_clob_*.jsonl`` under ``clob_dir``, newest file last."""
    if not clob_dir.is_dir():
        return []
    paths = sorted(clob_dir.glob("crypto_clob_*.jsonl"))
    merged: List[Dict[str, Any]] = []
    for p in paths:
        merged.extend(load_jsonl_snapshots(p))
    return merged


def family_key_to_split_filename(family_key: str) -> str:
    """
    Stable filename for **one JSONL per market family** (used by ``split_clob_snapshots_by_family``).

    Example: ``[15M] btc-updown-15m`` → ``15M_btc-updown-15m.jsonl``
    """
    s = (family_key or "").strip()
    s = re.sub(r"^\[([^\]]+)\]\s*", r"\1_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unknown"
    return f"{s[:200]}.jsonl"


def list_split_family_files(split_dir: Path) -> List[Path]:
    """Sorted ``*.jsonl`` paths under ``split_dir`` (one market family per file)."""
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*.jsonl"))


def load_split_family_files(paths: List[Path]) -> List[Dict[str, Any]]:
    """Merge rows from split-family JSONL files (same schema as monolithic CLOB snapshots)."""
    merged: List[Dict[str, Any]] = []
    for p in paths:
        merged.extend(load_jsonl_snapshots(p))
    return merged


def index_recorded_markets(rows: List[Dict[str, Any]]) -> Dict[str, RecordedMarketInfo]:
    """Group rows by ``condition_id`` (fallback: token pair) and return summary per market."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cid = (r.get("condition_id") or "").strip()
        if not cid:
            yt = (r.get("yes_token_id") or "").strip()
            nt = (r.get("no_token_id") or "").strip()
            cid = f"{yt}:{nt}" if yt and nt else ""
        if not cid:
            continue
        groups.setdefault(cid, []).append(r)

    out: Dict[str, RecordedMarketInfo] = {}
    for cid, g in groups.items():
        g_sorted = sorted(g, key=lambda x: parse_ts_utc(str(x.get("ts_utc") or "")))
        first = g_sorted[0]
        last = g_sorted[-1]
        out[cid] = RecordedMarketInfo(
            condition_id=cid,
            market_slug=str(first.get("market_slug") or ""),
            event_slug=str(first.get("event_slug") or ""),
            event_title=str(first.get("event_title") or ""),
            bucket=str(first.get("bucket") or ""),
            page_key=str(first.get("page_key") or ""),
            n_rows=len(g_sorted),
            t_first_iso=str(first.get("ts_utc") or ""),
            t_last_iso=str(last.get("ts_utc") or ""),
        )
    return out


def market_family_key(slug: str) -> str:
    """
    Strip a trailing window / epoch suffix so recurring markets group together.

    Example: ``sol-updown-15m-1774381500`` → ``sol-updown-15m``.
    """
    s = (slug or "").strip()
    if not s:
        return ""
    # Numeric epoch suffix style
    s = re.sub(r"-\d{9,}$", "", s)
    # Human datetime slug styles:
    #   "...-on-march-26-2026-4pm-et" / "...-on-march-26-2026-10am-et"
    #   "...-march-26-2026-4pm-et" / "...-march-26-4pm-et"
    #   "...-at-4pm-et"
    s = re.sub(r"-(on|at)-[a-z]+-\d{1,2}-\d{4}-\d{1,2}(am|pm)-et$", "", s)
    s = re.sub(r"-[a-z]+-\d{1,2}-\d{4}-\d{1,2}(am|pm)-et$", "", s)
    s = re.sub(r"-[a-z]+-\d{1,2}-\d{1,2}(am|pm)-et$", "", s)
    s = re.sub(r"-(on|at)-\d{1,2}(am|pm)-et$", "", s)
    return s or slug


def family_group_key(info: RecordedMarketInfo) -> str:
    """Human-readable key: bucket + de-windowed slug (or title fallback)."""
    # Use market_slug first: event_slug can encode specific window time (e.g. 4pm/5pm),
    # which would incorrectly split one recurring market type into multiple families.
    slug = (info.market_slug or info.event_slug or "").strip()
    core = market_family_key(slug)
    if not core:
        t = (info.event_title or "market")[:56]
        return f"[{info.bucket}] {t}"
    return f"[{info.bucket}] {core}"


def family_key_from_snapshot_row(row: Dict[str, Any]) -> str:
    """
    Same family label as ``family_group_key(RecordedMarketInfo)`` for a single JSONL / DB row.
    Used when indexing rows without building ``RecordedMarketInfo`` first.
    """
    slug = (str(row.get("market_slug") or "").strip() or str(row.get("event_slug") or "").strip())
    core = market_family_key(slug)
    bucket = str(row.get("bucket") or "")
    if not core:
        t = (str(row.get("event_title") or "market"))[:56]
        return f"[{bucket}] {t}"
    return f"[{bucket}] {core}"


def load_all_from_sqlite(db_path: Path) -> List[Dict[str, Any]]:
    """All snapshot rows from a SQLite store (see ``clob_snapshot_store``), chronological by ``ts_utc``."""
    from agents.application.clob_snapshot_store import load_rows_from_sqlite

    return load_rows_from_sqlite(db_path, family_keys=None)


def index_family_chains(
    idx: Dict[str, RecordedMarketInfo],
) -> Dict[str, List[str]]:
    """
    Map family key → ``condition_id``s in chronological order (first snapshot time).

    Each ``condition_id`` is one tradable window; the chain is successive resets
    of the same recurring product line.
    """
    buckets: Dict[str, List[str]] = {}
    for cid, info in idx.items():
        buckets.setdefault(family_group_key(info), []).append(cid)
    for k in list(buckets.keys()):
        buckets[k].sort(key=lambda c: parse_ts_utc(idx[c].t_first_iso))
    return buckets


def snapshot_row_yes_ask_bid(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    yb, ya = best_bid_ask(row.get("yes_book"))
    return yb, ya


def window_outcome_book_extremes(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """
    Per market window: extremes of **best bid** (low) and **best ask** (high) for YES and NO books.

    - ``yes_low`` / ``no_low``: minimum best bid seen across snapshots
    - ``yes_high`` / ``no_high``: maximum best ask seen across snapshots

    If a side has bids but no asks (or vice versa), the missing extreme is filled from the
    available quotes so callers can still plot a range.
    """
    yes_bids: List[float] = []
    yes_asks: List[float] = []
    no_bids: List[float] = []
    no_asks: List[float] = []
    for row in snapshots:
        yb, ya = best_bid_ask(row.get("yes_book"))
        nb, na = best_bid_ask(row.get("no_book"))
        if yb is not None:
            yes_bids.append(yb)
        if ya is not None:
            yes_asks.append(ya)
        if nb is not None:
            no_bids.append(nb)
        if na is not None:
            no_asks.append(na)

    def _low_high(bids: List[float], asks: List[float]) -> Tuple[Optional[float], Optional[float]]:
        lo = min(bids) if bids else None
        hi = max(asks) if asks else None
        if lo is None and hi is None:
            return None, None
        if lo is None:
            lo = hi
        if hi is None:
            hi = lo
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    y_lo, y_hi = _low_high(yes_bids, yes_asks)
    n_lo, n_hi = _low_high(no_bids, no_asks)
    return {
        "yes_low": y_lo,
        "yes_high": y_hi,
        "no_low": n_lo,
        "no_high": n_hi,
    }


def merged_yes_ask_series_for_chain(
    rows_all: List[Dict[str, Any]],
    condition_ids: List[str],
    *,
    break_between_windows: bool = True,
) -> Tuple[List[Optional[str]], List[Optional[float]], List[Optional[float]]]:
    """
    Concatenate YES bid/ask along wall-clock time for a chain of windows.

    Inserts ``None`` x/y pairs between windows so Plotly breaks the line.
    """
    xs: List[Optional[str]] = []
    y_bid: List[Optional[float]] = []
    y_ask: List[Optional[float]] = []
    first = True
    for cid in condition_ids:
        if not first and break_between_windows:
            xs.append(None)
            y_bid.append(None)
            y_ask.append(None)
        first = False
        for row in rows_for_condition(rows_all, cid):
            ts = str(row.get("ts_utc") or "")
            yb, ya = snapshot_row_yes_ask_bid(row)
            xs.append(ts)
            y_bid.append(yb)
            y_ask.append(ya)
    return xs, y_bid, y_ask


WindowSimRow = Dict[str, Any]


def backtest_family_chains(
    rows_all: List[Dict[str, Any]],
    idx: Dict[str, RecordedMarketInfo],
    family_keys: List[str],
    family_to_cids: Dict[str, List[str]],
    *,
    buy_price: float,
    sell_price: float,
    size: float,
    settle_each_window: Callable[[RecordedMarketInfo], Tuple[bool, bool]],
) -> Dict[str, Any]:
    """
    Run one fresh dual-limit simulation per ``condition_id`` (each market window),
    then aggregate. ``settle_each_window(info)`` → ``(yes_expires_at_1, apply_settlement)``.

    Capital is **re-deployed** each window: total deployed = per-window deployed × windows.
    """
    per_window: List[WindowSimRow] = []
    total_cash = 0.0
    total_deployed = 0.0

    for fam in family_keys:
        cids = family_to_cids.get(fam) or []
        for cid in cids:
            info = idx[cid]
            snaps = rows_for_condition(rows_all, cid)
            if len(snaps) < 1:
                continue
            yes_1, apply_set = settle_each_window(info)
            sim = simulate_dual_limit_book_snapshots(
                snaps,
                buy_price=buy_price,
                sell_price=sell_price,
                size=size,
                yes_expires_at_1=yes_1,
                apply_settlement=apply_set,
            )
            pnl = (
                float(sim["cash_after_settlement"])
                if apply_set
                else float(sim["cash_after_trades"])
            )
            dep = float(sim["deployed_capital_usdc"])
            total_cash += pnl
            total_deployed += dep
            per_window.append(
                {
                    "family": fam,
                    "condition_id": cid[:20] + "…" if len(cid) > 24 else cid,
                    "market_slug": info.market_slug,
                    "n_snapshots": len(snaps),
                    "pnl_usdc": round(pnl, 6),
                    "deployed_usdc": dep,
                    "bought_yes": sim["bought_yes"],
                    "bought_no": sim["bought_no"],
                    "sold_yes": sim["sold_yes"],
                    "sold_no": sim["sold_no"],
                }
            )

    agg_ret = (100.0 * total_cash / total_deployed) if total_deployed > 0 else 0.0
    return {
        "per_window": per_window,
        "total_pnl_usdc": round(total_cash, 6),
        "total_deployed_usdc": round(total_deployed, 6),
        "aggregate_return_pct": round(agg_ret, 4),
        "n_windows": len(per_window),
    }


def condition_key_from_row(r: Dict[str, Any]) -> str:
    """Stable id: ``condition_id`` or ``yes_token_id:no_token_id``."""
    cid = (r.get("condition_id") or "").strip()
    if cid:
        return cid
    yt = (r.get("yes_token_id") or "").strip()
    nt = (r.get("no_token_id") or "").strip()
    return f"{yt}:{nt}" if yt and nt else ""


def group_rows_by_condition(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Single-pass index: ``condition_id`` → chronologically sorted snapshot rows.

    Use this instead of calling ``rows_for_condition`` in a loop over windows (avoids O(n×windows)).
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cid = condition_key_from_row(r)
        if not cid:
            continue
        groups.setdefault(cid, []).append(r)
    for g in groups.values():
        g.sort(key=lambda x: parse_ts_utc(str(x.get("ts_utc") or "")))
    return groups


def rows_for_condition(rows: List[Dict[str, Any]], condition_id: str) -> List[Dict[str, Any]]:
    sel = [r for r in rows if condition_key_from_row(r) == condition_id]
    return sorted(sel, key=lambda x: parse_ts_utc(str(x.get("ts_utc") or "")))


def simulate_dual_limit_book_snapshots(
    snapshots: List[Dict[str, Any]],
    *,
    buy_price: float,
    sell_price: float,
    size: float,
    yes_expires_at_1: bool,
    apply_settlement: bool = True,
) -> Dict[str, Any]:
    """
    Same economic intent as ``simulate_one_shot_dual_limit`` but uses best bid/ask per snapshot.
    """
    pos_y = 0.0
    pos_n = 0.0
    cash = 0.0
    bought_y = bought_n = False
    sold_y = sold_n = False
    fill_buy_y_ask: Optional[float] = None
    fill_buy_n_ask: Optional[float] = None
    fill_sell_y_bid: Optional[float] = None
    fill_sell_n_bid: Optional[float] = None
    series: List[Dict[str, Any]] = []

    for row in snapshots:
        ts = str(row.get("ts_utc") or "")
        yb, ya = best_bid_ask(row.get("yes_book"))
        nb, na = best_bid_ask(row.get("no_book"))
        series.append(
            {
                "ts_utc": ts,
                "yes_bid": yb,
                "yes_ask": ya,
                "no_bid": nb,
                "no_ask": na,
            }
        )

        if ya is not None and not bought_y and ya <= buy_price:
            cash -= ya * size
            pos_y += size
            bought_y = True
            fill_buy_y_ask = ya
        if na is not None and not bought_n and na <= buy_price:
            cash -= na * size
            pos_n += size
            bought_n = True
            fill_buy_n_ask = na
        if pos_y > 0 and not sold_y and yb is not None and yb >= sell_price:
            cash += yb * pos_y
            pos_y = 0.0
            sold_y = True
            fill_sell_y_bid = yb
        if pos_n > 0 and not sold_n and nb is not None and nb >= sell_price:
            cash += nb * pos_n
            pos_n = 0.0
            sold_n = True
            fill_sell_n_bid = nb

    cash_after_trades = cash
    if apply_settlement:
        if pos_y > 0:
            cash += (1.0 if yes_expires_at_1 else 0.0) * pos_y
        if pos_n > 0:
            cash += (0.0 if yes_expires_at_1 else 1.0) * pos_n

    deployed = 2.0 * buy_price * size
    return {
        "cash_after_trades": round(cash_after_trades, 6),
        "cash_after_settlement": round(cash, 6),
        "deployed_capital_usdc": round(deployed, 6),
        "return_on_capital_pct": round(
            (100.0 * cash / deployed) if deployed > 0 else 0.0,
            4,
        ),
        "bought_yes": bought_y,
        "bought_no": bought_n,
        "sold_yes": sold_y,
        "sold_no": sold_n,
        "pos_yes_open": pos_y,
        "pos_no_open": pos_n,
        "fill_buy_yes_ask": fill_buy_y_ask,
        "fill_buy_no_ask": fill_buy_n_ask,
        "fill_sell_yes_bid": fill_sell_y_bid,
        "fill_sell_no_bid": fill_sell_n_bid,
        "series": series,
        "note": "Book touch fills; no fees; settlement from caller-provided outcome.",
    }
