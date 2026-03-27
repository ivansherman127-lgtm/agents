"""
Historical outcome (share) prices from Polymarket's CLOB `prices-history` API.

Use this to backtest strategies per market window. The API returns sampled mids
(`p` in USD probability terms) for each outcome **token id** — not the condition id.

The CLOB query param ``fidelity`` is an interval in **minutes** (fractional allowed).
We take ``sample_interval_ms`` from callers and convert: ``fidelity = ms / 60_000``.
For ~15-minute markets the API still caps how many points you get (often about
one sample per minute regardless of very small intervals).

Docs mirror py-clob-client read-only usage: no auth required for this endpoint.

**More accurate data for tight strategies (fills / bid–ask / tape):**

- ``prices-history`` is **not** tick-level and is not guaranteed to match tradable
  bid/ask; it is a coarse **mid** series.
- **Current** full book: ``GET /book?token_id=...`` (no auth) while the market is
  open — snapshot only; there is no public historical order-book archive. You can
  **poll** and store snapshots yourself (or in the bot) to build a ground-truth
  series for live windows.
- **Last trade:** ``GET /last-trade-price?token_id=...`` — point-in-time, not history.
- **Trade history** via ``GET /data/trades`` (``py_clob_client`` ``get_trades``)
  requires **Level 2 CLOB API credentials**; params support ``asset_id`` / ``market``
  and ``before`` / ``after`` for pagination. That yields **actual executed**
  prices/sizes for trades the API returns (often oriented to your account — check
  Polymarket’s current docs for market-wide vs user-scoped access).
- **On-chain:** conditional-token / CTF exchange events on Polygon are in principle
  reconstructible (subgraph, indexer, ``eth_getLogs``) for true settlement and
  fills, but that is separate from the Gamma/CLOB convenience APIs.

For variation-sensitive backtests, prefer **recording book or trade tape yourself**
while the strategy runs, or authenticated trade export + reconstruct **limit fill**
assumptions from best bid/ask at each poll.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
DEFAULT_UA = "Mozilla/5.0 (compatible; PolymarketBacktest/1.0)"


def _http_get_json(url: str, timeout_s: float = 60.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iso8601_to_unix(iso: str) -> int:
    """Parse Gamma-style ISO8601 (Z suffix) to unix seconds UTC."""
    if not iso:
        return 0
    s = str(iso).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_gamma_market_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    slug = slug.strip()
    if not slug:
        return None
    q = urllib.parse.urlencode({"slug": slug})
    url = f"{GAMMA_API}/markets?{q}"
    try:
        data = _http_get_json(url)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list) and data:
        return data[0]
    return None


def parse_clob_token_ids(market: Dict[str, Any]) -> Tuple[str, str]:
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("Market is not a binary clob market (need two token ids)")
    return str(raw[0]), str(raw[1])


def window_unix_from_gamma_market(market: Dict[str, Any]) -> Tuple[int, int]:
    """Trading window [start, end] in unix seconds from `eventStartTime` / `endDate`."""
    start_s = iso8601_to_unix(str(market.get("eventStartTime") or ""))
    end_s = iso8601_to_unix(str(market.get("endDate") or ""))
    if start_s <= 0 or end_s <= 0 or start_s >= end_s:
        raise ValueError("Market missing eventStartTime/endDate or invalid range")
    return start_s, end_s


def sample_interval_ms_to_fidelity_str(sample_interval_ms: int) -> str:
    """Convert UI milliseconds to CLOB ``fidelity`` (minutes)."""
    if sample_interval_ms < 1:
        raise ValueError("sample_interval_ms must be >= 1")
    minutes = sample_interval_ms / 60_000.0
    # Compact float string for URL (API accepts fractional minutes).
    s = f"{minutes:.12g}"
    return s


def coalesce_sample_interval_ms(
    *,
    sample_interval_ms: Optional[int] = None,
    fidelity_minutes: Optional[float] = None,
) -> int:
    """
    Resolve sampling interval: prefer ``sample_interval_ms``, else legacy ``fidelity_minutes`` × 60_000.
    """
    if sample_interval_ms is not None and fidelity_minutes is not None:
        raise TypeError("Pass only one of sample_interval_ms or fidelity_minutes")
    if sample_interval_ms is not None:
        ms = int(sample_interval_ms)
        if ms < 1:
            raise ValueError("sample_interval_ms must be >= 1")
        return ms
    if fidelity_minutes is not None:
        return max(1, int(round(float(fidelity_minutes) * 60_000.0)))
    return 60_000


def fetch_prices_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    *,
    sample_interval_ms: Optional[int] = None,
    fidelity_minutes: Optional[float] = None,
    host: str = CLOB_HOST,
    timeout_s: float = 60.0,
) -> List[Dict[str, Any]]:
    """
    GET /prices-history

    `market` query param is the **CLOB outcome token id** (same as ``token_id`` here).
    Pass ``sample_interval_ms`` (or deprecated ``fidelity_minutes``); mapped to CLOB ``fidelity`` in minutes.

    Returns rows ``{"t": unix_seconds, "p": float}`` sorted by time (API order).
    """
    if start_ts >= end_ts:
        raise ValueError("start_ts must be < end_ts")
    ms = coalesce_sample_interval_ms(
        sample_interval_ms=sample_interval_ms,
        fidelity_minutes=fidelity_minutes,
    )
    params = {
        "market": str(token_id),
        "startTs": str(int(start_ts)),
        "endTs": str(int(end_ts)),
        "fidelity": sample_interval_ms_to_fidelity_str(ms),
    }
    url = f"{host}/prices-history?" + urllib.parse.urlencode(params)
    data = _http_get_json(url, timeout_s=timeout_s)
    if not isinstance(data, dict):
        return []
    hist = data.get("history")
    if not isinstance(hist, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in hist:
        if not isinstance(row, dict):
            continue
        try:
            t = int(row["t"])
            p = float(row["p"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"t": t, "p": p})
    out.sort(key=lambda r: r["t"])
    return out


def forward_filled_merge(
    yes_hist: List[Dict[str, Any]],
    no_hist: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build rows ``{t, yes_p, no_p}`` at every distinct timestamp, forward-filling
    the last known price for each leg (CLOB samples may not align).
    """
    yes_pts = [(int(r["t"]), float(r["p"])) for r in yes_hist]
    no_pts = [(int(r["t"]), float(r["p"])) for r in no_hist]
    all_t = sorted({t for t, _ in yes_pts} | {t for t, _ in no_pts})
    if not all_t:
        return []

    yes_pts.sort(key=lambda x: x[0])
    no_pts.sort(key=lambda x: x[0])
    yi = ni = 0
    last_y: Optional[float] = None
    last_n: Optional[float] = None
    rows: List[Dict[str, Any]] = []
    for t in all_t:
        while yi < len(yes_pts) and yes_pts[yi][0] <= t:
            last_y = yes_pts[yi][1]
            yi += 1
        while ni < len(no_pts) and no_pts[ni][0] <= t:
            last_n = no_pts[ni][1]
            ni += 1
        rows.append({"t": t, "yes_p": last_y, "no_p": last_n})
    return rows


@dataclass
class BinaryWindowPriceBundle:
    slug: str
    condition_id: str
    yes_token_id: str
    no_token_id: str
    window_start_unix: int
    window_end_unix: int
    question: str
    yes_history: List[Dict[str, Any]]
    no_history: List[Dict[str, Any]]
    aligned: List[Dict[str, Any]]

    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "condition_id": self.condition_id,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "window_start_unix": self.window_start_unix,
            "window_end_unix": self.window_end_unix,
            "question": self.question,
            "yes_history": self.yes_history,
            "no_history": self.no_history,
            "aligned": self.aligned,
        }


def fetch_binary_window_prices(
    slug: str,
    *,
    sample_interval_ms: Optional[int] = None,
    fidelity_minutes: Optional[float] = None,
    pad_seconds: int = 120,
) -> BinaryWindowPriceBundle:
    """
    Resolve slug via Gamma, fetch YES/NO ``prices-history`` over the event window
    (optionally padded for boundary samples).

    Pass ``sample_interval_ms`` (recommended) or deprecated ``fidelity_minutes`` (converted to ms).
    """
    m = fetch_gamma_market_by_slug(slug)
    if not m:
        raise RuntimeError(f"No Gamma market for slug={slug!r}")
    yes_id, no_id = parse_clob_token_ids(m)
    w0, w1 = window_unix_from_gamma_market(m)
    start_ts = w0 - max(0, pad_seconds)
    end_ts = w1 + max(0, pad_seconds)
    ms = coalesce_sample_interval_ms(
        sample_interval_ms=sample_interval_ms,
        fidelity_minutes=fidelity_minutes,
    )
    yh = fetch_prices_history(yes_id, start_ts, end_ts, sample_interval_ms=ms)
    nh = fetch_prices_history(no_id, start_ts, end_ts, sample_interval_ms=ms)
    aligned = forward_filled_merge(yh, nh)
    return BinaryWindowPriceBundle(
        slug=str(m.get("slug") or slug),
        condition_id=str(m.get("conditionId") or ""),
        yes_token_id=yes_id,
        no_token_id=no_id,
        window_start_unix=w0,
        window_end_unix=w1,
        question=str(m.get("question") or ""),
        yes_history=yh,
        no_history=nh,
        aligned=aligned,
    )


def simulate_one_shot_dual_limit(
    aligned: List[Dict[str, Any]],
    *,
    buy_price: float,
    sell_price: float,
    size: float,
    # Optional final settlement: 1 = YES/Up pays $1, 0 = NO/Down pays $1 on NO leg (model: YES token 1, NO token 0)
    yes_expires_at_1: bool,
) -> Dict[str, Any]:
    """
    Crude backtest using **mid** samples as fillability proxies (not real book).

    Rules:
    - BUY leg when mid ``<= buy_price`` (limit buy would execute if ask is at/below).
    - SELL leg when mid ``>= sell_price`` after the buy is counted.
    - Settlement: ``yes_expires_at_1`` means YES shares worth 1 and NO worth 0; else opposite.
    """
    pos_y = pos_n = 0.0
    cash = 0.0
    bought_y = bought_n = False
    sold_y = sold_n = False

    for row in aligned:
        yp = row.get("yes_p")
        np_ = row.get("no_p")
        if yp is not None and not bought_y and float(yp) <= buy_price:
            cash -= buy_price * size
            pos_y += size
            bought_y = True
        if np_ is not None and not bought_n and float(np_) <= buy_price:
            cash -= buy_price * size
            pos_n += size
            bought_n = True
        if pos_y > 0 and not sold_y and yp is not None and float(yp) >= sell_price:
            cash += sell_price * pos_y
            pos_y = 0.0
            sold_y = True
        if pos_n > 0 and not sold_n and np_ is not None and float(np_) >= sell_price:
            cash += sell_price * pos_n
            pos_n = 0.0
            sold_n = True

    if pos_y > 0:
        cash += (1.0 if yes_expires_at_1 else 0.0) * pos_y
    if pos_n > 0:
        cash += (0.0 if yes_expires_at_1 else 1.0) * pos_n

    return {
        "cash_after_settlement": round(cash, 6),
        "bought_yes": bought_y,
        "bought_no": bought_n,
        "sold_yes": sold_y,
        "sold_no": sold_n,
        "note": "Mid-price proxy; not fill-accurate vs real LOB or fees.",
    }
