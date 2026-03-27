from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_session_file_stem(session_id: str) -> str:
    """Filesystem-safe stem matching JSONL session filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)


@dataclass
class TradeSessionMetrics:
    """Append-only session log for dashboard / post-mortem analysis."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_slug: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Optional[str] = None
    sessions_dir: str = "data/sessions"
    _jsonl_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self._jsonl_path:
            os.makedirs(self.sessions_dir, exist_ok=True)
            safe_id = safe_session_file_stem(self.session_id)
            self._jsonl_path = os.path.join(self.sessions_dir, f"{safe_id}.jsonl")

    def record(self, event_type: str, **fields: Any) -> None:
        row = {
            "type": event_type,
            "ts_monotonic": time.monotonic(),
            "ts_utc": utc_iso(),
            **fields,
        }
        self.events.append(row)
        if self._jsonl_path:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")

    def set_outcome(self, outcome: str) -> None:
        self.outcome = outcome
        self.record("outcome", outcome=outcome)

    def summary(self) -> Dict[str, Any]:
        buy_price = sell_price = size = None
        est_pnl = 0.0
        for e in self.events:
            if e["type"] == "placed_buy":
                buy_price = e.get("price")
                size = e.get("size")
            if e["type"] == "placed_sell":
                sell_price = e.get("price")
                sz = e.get("size", size)
                bp = e.get("buy_price", buy_price)
                if (
                    bp is not None
                    and sell_price is not None
                    and sz is not None
                ):
                    est_pnl += (float(sell_price) - float(bp)) * float(sz)

        ttf_yes = self._time_to_full_fill("YES")
        ttf_no = self._time_to_full_fill("NO")
        poll_count = sum(1 for e in self.events if e["type"] == "poll_iteration")

        return {
            "session_id": self.session_id,
            "market_slug": self.market_slug,
            "outcome": self.outcome,
            "estimated_pnl_usdc_ex_fees": round(est_pnl, 6),
            "poll_iterations": poll_count,
            "time_to_full_fill_seconds": {"YES": ttf_yes, "NO": ttf_no},
        }

    def _time_to_full_fill(self, side: str) -> Optional[float]:
        t_buy = None
        t_full = None
        target = None
        for e in self.events:
            if e["type"] == "placed_buy" and e.get("side") == side:
                t_buy = e["ts_monotonic"]
                target = float(e.get("size", 0))
            if e["type"] == "wait_fill" and e.get("side") == side:
                filled = float(e.get("filled", 0))
                if target and filled >= target and t_full is None:
                    t_full = e["ts_monotonic"]
        if t_buy is not None and t_full is not None:
            return round(t_full - t_buy, 4)
        return None


def extract_order_id_from_response(order_response: Any) -> str:
    """Same logic as Polymarket.extract_order_id without importing Polymarket."""
    if isinstance(order_response, str):
        return order_response
    if not isinstance(order_response, dict):
        return ""
    for key in ("orderID", "orderId", "id"):
        if key in order_response and order_response[key]:
            return str(order_response[key])
    if "data" in order_response and isinstance(order_response["data"], dict):
        nested = order_response["data"]
        for key in ("orderID", "orderId", "id"):
            if key in nested and nested[key]:
                return str(nested[key])
    return ""
