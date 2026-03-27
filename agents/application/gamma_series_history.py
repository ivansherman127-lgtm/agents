"""
Fetch recurring Polymarket event windows (same series) from Gamma API and parse resolutions.

Uses stdlib HTTP only so callers (e.g. Streamlit) need not import trading stack.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GAMMA_API = "https://gamma-api.polymarket.com"
DEFAULT_UA = "Mozilla/5.0 (compatible; PolymarketMetrics/1.0)"


def _http_get_json(url: str, timeout_s: float = 45.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def window_ts_from_slug(slug: str) -> Optional[int]:
    """Unix second suffix shared by e.g. eth-updown-15m-* and eth-up-or-down-15m-*."""
    slug = (slug or "").strip()
    m = re.search(r"-(\d{10,})$", slug)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_json_list_field(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def _canonical_ud_label(name: str) -> str:
    t = str(name).strip().lower()
    if t == "up":
        return "Up"
    if t == "down":
        return "Down"
    return str(name)


def resolved_winner_outcome_label(market: Dict[str, Any]) -> Optional[str]:
    """
    Winning outcome label (e.g. Up / Down), aligned with Polymarket's displayed prices.

    Gamma often keeps ``closed`` false briefly while ``outcomePrices`` are already
    ~100% / 0% (same as the website). We therefore treat two-outcome markets as
    settled when max price is high and min is low, without requiring ``closed``.
    """
    outcomes = _parse_json_list_field(market.get("outcomes"))
    prices = _parse_json_list_field(market.get("outcomePrices"))
    if len(outcomes) != len(prices) or not outcomes:
        return None

    fps: List[float] = []
    for p in prices:
        try:
            fps.append(float(p))
        except (TypeError, ValueError):
            return None

    n = len(fps)
    # Match UI: ~resolved when one side is near 1 and the other near 0 (even if `closed` is still false).
    if n == 2:
        hi, lo = max(fps), min(fps)
        if hi >= 0.95 and lo <= 0.05:
            widx = fps.index(hi)
            return _canonical_ud_label(str(outcomes[widx]))

    if market.get("closed"):
        for name, fp in zip(outcomes, fps):
            if fp >= 0.99:
                return _canonical_ud_label(str(name))
        widx = max(range(n), key=lambda i: fps[i])
        if fps[widx] >= 0.5:
            return _canonical_ud_label(str(outcomes[widx]))
    return None


def series_id_from_event(event: Dict[str, Any]) -> Optional[str]:
    series = event.get("series")
    if isinstance(series, list) and series:
        sid = series[0].get("id")
        if sid is not None:
            return str(sid)
    return None


def event_end_datetime(ev: Dict[str, Any]) -> datetime:
    """Primary market end time for sorting (UTC). Missing dates sort as minimum."""
    mk = ev.get("markets")
    raw = ""
    if isinstance(mk, list) and mk:
        raw = str(mk[0].get("endDate") or ev.get("endDate") or "")
    else:
        raw = str(ev.get("endDate") or "")
    if not raw:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def sort_series_events_newest_first(events: List[Dict[str, Any]]) -> None:
    """In-place sort by market end time descending (defensive if API order drifts)."""
    events.sort(key=event_end_datetime, reverse=True)


def fetch_event_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """GET /events?slug=… — first match or None."""
    slug = slug.strip()
    if not slug:
        return None
    q = urllib.parse.urlencode({"slug": slug})
    url = f"{GAMMA_API}/events?{q}"
    try:
        data = _http_get_json(url)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list) and data:
        return data[0]
    return None


def fetch_series_events(
    series_id: str,
    *,
    limit_total: int = 400,
    page_size: int = 100,
    order: str = "endDate",
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """
    Paginate GET /events?series_id=… with ``ascending=False`` (newest ``endDate`` first).

    The API must receive ``ascending=false``; otherwise Gamma returns the **oldest**
    windows first (e.g. September 2025 for ETH 15m), which is not what we want.

    After pagination, results are sorted again by end time descending as a safeguard.
    """
    out: List[Dict[str, Any]] = []
    offset = 0
    series_id = str(series_id).strip()
    while len(out) < limit_total:
        n = min(page_size, limit_total - len(out))
        params = {
            "series_id": series_id,
            "limit": str(n),
            "offset": str(offset),
            "order": order,
            "ascending": "true" if ascending else "false",
        }
        q = urllib.parse.urlencode(params)
        url = f"{GAMMA_API}/events?{q}"
        try:
            chunk = _http_get_json(url)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            break
        if not isinstance(chunk, list) or not chunk:
            break
        out.extend(chunk)
        if len(chunk) < n:
            break
        offset += len(chunk)

    sort_series_events_newest_first(out)
    if len(out) > limit_total:
        out[:] = out[:limit_total]
    return out


@dataclass
class SeriesWindowRow:
    slug: str
    end_date_iso: str
    end_dt: Optional[datetime]
    resolution: Optional[str]
    market_closed: bool
    title: str


def event_primary_market(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    markets = event.get("markets")
    if isinstance(markets, list) and markets:
        return markets[0]
    return None


def window_row_from_event(event: Dict[str, Any]) -> Optional[SeriesWindowRow]:
    mk = event_primary_market(event)
    if not mk:
        return None
    slug = str(event.get("slug") or "")
    end_raw = str(mk.get("endDate") or event.get("endDate") or "")
    end_dt: Optional[datetime] = None
    if end_raw:
        try:
            end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
        except ValueError:
            end_dt = None
    return SeriesWindowRow(
        slug=slug,
        end_date_iso=end_raw,
        end_dt=end_dt,
        resolution=resolved_winner_outcome_label(mk),
        market_closed=bool(mk.get("closed")),
        title=str(event.get("title") or mk.get("question") or slug),
    )


def build_window_index(rows: List[SeriesWindowRow]) -> Dict[int, SeriesWindowRow]:
    """Map unix window suffix -> row (last wins if duplicates)."""
    idx: Dict[int, SeriesWindowRow] = {}
    for r in rows:
        ts = window_ts_from_slug(r.slug)
        if ts is not None:
            idx[ts] = r
    return idx
