from __future__ import annotations

from typing import Any, Dict, Literal, Set

from agents.application.session_metrics import extract_order_id_from_response

FillMode = Literal["instant", "delayed", "never"]


class DryRunPolymarket:
    """
    Simulates Polymarket order placement and fills for strategy dry-runs.
    No CLOB or HTTP calls.
    """

    def __init__(
        self,
        fill_mode: FillMode = "instant",
        delay_polls: int = 2,
        market_open: bool = True,
    ) -> None:
        self.fill_mode = fill_mode
        self.delay_polls = max(1, delay_polls)
        self._market_open = market_open
        self._order_counter = 0
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._fill_checks: Dict[str, int] = {}
        self._cancelled: Set[str] = set()

    def execute_order(self, price: float, size: float, side: str, token_id: str) -> dict:
        self._order_counter += 1
        order_id = f"dry-{side.lower()}-{self._order_counter}"
        self._orders[order_id] = {
            "price": price,
            "size": size,
            "side": side,
            "token_id": token_id,
        }
        self._fill_checks[order_id] = 0
        return {"orderID": order_id}

    def cancel_order(self, order_id: str) -> dict:
        meta = self._orders.get(order_id)
        ff = 0.0
        if meta and meta.get("side") == "BUY":
            target = float(meta["size"])
            if self.fill_mode == "instant":
                ff = target
            elif self.fill_mode == "never":
                ff = 0.0
            else:
                n = self._fill_checks.get(order_id, 0)
                ff = target if n >= self.delay_polls else 0.0
            meta["frozen_fill"] = ff
        if meta:
            meta["cancelled"] = True
        self._cancelled.add(order_id)
        return {"canceled": True, "orderID": order_id}

    def extract_order_id(self, order_response: Any) -> str:
        return extract_order_id_from_response(order_response)

    def get_order_filled_size(self, order_id: str) -> float:
        meta = self._orders.get(order_id)
        if not meta:
            return 0.0
        if meta.get("cancelled") and meta.get("side") == "BUY":
            return float(meta.get("frozen_fill", 0.0))
        target = float(meta["size"])
        if meta["side"] != "BUY":
            return target

        if self.fill_mode == "never":
            return 0.0
        if self.fill_mode == "instant":
            return target

        self._fill_checks[order_id] = self._fill_checks.get(order_id, 0) + 1
        n = self._fill_checks[order_id]
        if n < self.delay_polls:
            return 0.0
        return target

    def is_market_open_by_slug(self, market_slug: str) -> bool:
        return self._market_open
