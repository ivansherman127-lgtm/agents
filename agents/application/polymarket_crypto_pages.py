"""
Discover crypto Up/Down (and related) markets listed on Polymarket /crypto/* hub pages.

Data comes from the embedded ``__NEXT_DATA__`` payload (same source the web app uses).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_UA = "Mozilla/5.0 (compatible; ClobSnapshotRecorder/1.0; +https://polymarket.com)"

CRYPTO_PAGE_PATHS = {
    "15M": "15M",
    "hourly": "hourly",
    "4hour": "4hour",
}

# Short label stored on each snapshot row
BUCKET_TAG = {
    "15M": "15M",
    "hourly": "1H",
    "4hour": "4H",
}


@dataclass(frozen=True)
class CryptoMarketTarget:
    """One binary CLOB market to snapshot (YES/NO token pair)."""

    bucket: str
    page_key: str
    event_slug: str
    event_title: str
    market_slug: str
    condition_id: str
    yes_token_id: str
    no_token_id: str


def _http_get(url: str, timeout_s: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_next_data(url: str) -> Dict[str, Any]:
    html = _http_get(url)
    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not m:
        raise RuntimeError(f"No __NEXT_DATA__ in page: {url}")
    return json.loads(m.group(1))


def _parse_clob_pair(market: Dict[str, Any]) -> Optional[tuple[str, str]]:
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return str(raw[0]), str(raw[1])


def _events_from_crypto_markets_query(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = state.get("data") or {}
    pages = data.get("pages") or []
    if not pages:
        return []
    return list(pages[0].get("events") or [])


def _events_from_four_hour_query(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = state.get("data")
    if isinstance(data, list):
        return list(data)
    return []


def targets_for_page(page_key: str) -> List[CryptoMarketTarget]:
    """
    ``page_key`` is ``15M``, ``hourly``, or ``4hour`` (URL path segment).
    """
    path = CRYPTO_PAGE_PATHS.get(page_key)
    if not path:
        raise ValueError(f"Unknown page_key={page_key!r}")
    url = f"https://polymarket.com/crypto/{path}"
    root = fetch_next_data(url)
    bucket = BUCKET_TAG[page_key]
    queries = (
        root.get("props", {})
        .get("pageProps", {})
        .get("dehydratedState", {})
        .get("queries", [])
    )
    events: List[Dict[str, Any]] = []
    for q in queries:
        qk = q.get("queryKey") or []
        if not qk:
            continue
        st = q.get("state") or {}
        if qk[0] == "crypto-markets":
            events.extend(_events_from_crypto_markets_query(st))
        elif qk[0] == "4-hour-markets":
            events.extend(_events_from_four_hour_query(st))

    out: List[CryptoMarketTarget] = []
    for ev in events:
        ev_slug = str(ev.get("slug") or "")
        ev_title = str(ev.get("title") or "")
        for mk in ev.get("markets") or []:
            if not isinstance(mk, dict):
                continue
            pair = _parse_clob_pair(mk)
            if not pair:
                continue
            yes_id, no_id = pair
            m_slug = str(mk.get("slug") or "")
            cond = str(mk.get("conditionId") or "")
            out.append(
                CryptoMarketTarget(
                    bucket=bucket,
                    page_key=page_key,
                    event_slug=ev_slug,
                    event_title=ev_title,
                    market_slug=m_slug,
                    condition_id=cond,
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                )
            )
    return out


def all_hub_targets(
    page_keys: Optional[List[str]] = None,
) -> List[CryptoMarketTarget]:
    keys = page_keys or list(CRYPTO_PAGE_PATHS.keys())
    seen: set = set()
    merged: List[CryptoMarketTarget] = []
    for pk in keys:
        for t in targets_for_page(pk):
            dedupe = t.condition_id or f"{t.yes_token_id}:{t.no_token_id}"
            if dedupe in seen:
                continue
            seen.add(dedupe)
            merged.append(t)
    return merged


def iter_all_hub_targets() -> Iterator[CryptoMarketTarget]:
    yield from all_hub_targets()
