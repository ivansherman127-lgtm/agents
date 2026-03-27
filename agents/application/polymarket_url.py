"""Parse Polymarket website URLs into gamma API slugs and clob token ids."""

from __future__ import annotations

from urllib.parse import unquote, urlparse


def slug_from_polymarket_url(url: str) -> str:
    """
    Extract a market/event slug from common Polymarket URLs, e.g.
    https://polymarket.com/event/will-it-rain
    """
    raw = url.strip()
    if not raw:
        raise ValueError("Empty market URL")
    parsed = urlparse(raw)
    path = unquote(parsed.path).strip("/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise ValueError(f"No path segment found in URL: {url!r}")
    for i, p in enumerate(parts):
        if p in ("event", "market") and i + 1 < len(parts):
            return parts[i + 1]
    return parts[-1]


def resolve_polymarket_market_url(url: str) -> tuple[str, str, str]:
    """
    Resolve a browser URL to YES token id, NO token id, and canonical market slug.

    Uses one Gamma API lookup + :class:`~agents.polymarket.polymarket.Polymarket`.
    """
    from agents.polymarket.polymarket import Polymarket

    slug = slug_from_polymarket_url(url)
    yes_id, no_id, canonical = Polymarket().get_binary_clob_token_ids_by_slug(slug)
    return yes_id, no_id, canonical
