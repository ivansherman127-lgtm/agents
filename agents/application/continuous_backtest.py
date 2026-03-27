"""
Append-only JSONL logs for scheduled hedge-cancel sweeps (and similar) over recorded CLOB data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.application.clob_snapshot_backtest import (
    index_family_chains,
    index_recorded_markets,
    load_all_snapshot_files,
    rows_for_condition,
)
from agents.application.hedge_cancel_strategy import sweep_hedge_cancel_margin_sliding
from agents.application.clob_price_history import fetch_gamma_market_by_slug
from agents.application.gamma_series_history import resolved_winner_outcome_label


def _settlement_yes_at_1_for_slug(market_slug: str) -> tuple[bool, Optional[str]]:
    m = fetch_gamma_market_by_slug(market_slug) if market_slug else None
    if not m:
        return True, None
    w = resolved_winner_outcome_label(m)
    if w == "Up":
        return True, w
    if w == "Down":
        return False, w
    return True, w


def run_hedge_cancel_sweep_log_cycle(
    clob_dir: Path,
    out_path: Path,
    *,
    family_keys: Optional[List[str]] = None,
    yes_buy_1: float = 0.45,
    yes_buy_2: float = 0.45,
    no_buy_2: float = 0.45,
    no_buy_min: float = 0.45,
    no_buy_max: float = 0.60,
    margin_cents_min: int = 1,
    margin_cents_max: int = 5,
    size: float = 5.0,
    apply_settlement: bool = True,
    n_cycles: int = 1,
    cycle_step_cents: int = 0,
    fill_policy: str = "limit",
) -> int:
    """
    For each market window in each family chain, run **sliding margin** sweep (same spread in
    dollars on **both** YES and NO: sell = buy + margin) × NO buy price range; append one JSON line per window.

    Returns number of rows written.
    """
    rows_all = load_all_snapshot_files(clob_dir)
    if not rows_all:
        return 0
    idx = index_recorded_markets(rows_all)
    chains = index_family_chains(idx)
    keys = family_keys if family_keys else sorted(chains.keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with out_path.open("a", encoding="utf-8") as f:
        for fk in keys:
            cids = chains.get(fk) or []
            for cid in cids:
                info = idx[cid]
                snaps = rows_for_condition(rows_all, cid)
                if len(snaps) < 1:
                    continue
                yes_1, winner = _settlement_yes_at_1_for_slug(info.market_slug)
                grid, global_best, best_per_margin = sweep_hedge_cancel_margin_sliding(
                    snaps,
                    yes_buy_1=yes_buy_1,
                    yes_buy_2=yes_buy_2,
                    no_buy_2=no_buy_2,
                    no_buy_min=no_buy_min,
                    no_buy_max=no_buy_max,
                    margin_cents_min=margin_cents_min,
                    margin_cents_max=margin_cents_max,
                    size=size,
                    yes_expires_at_1=yes_1,
                    apply_settlement=apply_settlement,
                    n_cycles=n_cycles,
                    cycle_step_cents=cycle_step_cents,
                    fill_policy=fill_policy,
                )
                rec: Dict[str, Any] = {
                    "kind": "hedge_cancel_sweep",
                    "sweep_mode": "margin_sliding",
                    "margin_cents_range": [margin_cents_min, margin_cents_max],
                    "n_cycles": n_cycles,
                    "cycle_step_cents": cycle_step_cents,
                    "fill_policy": fill_policy,
                    "ts_utc": ts,
                    "strategy": "hedge_cancel",
                    "strategy_label": (
                        "Hedge-cancel (symmetric): buys on YES & NO; first leg to finish buy+sell "
                        "→ cancel other buy if unfilled; else round 2 on the cancelled side"
                    ),
                    "family": fk,
                    "condition_id": cid,
                    "market_slug": info.market_slug,
                    "n_snapshots": len(snaps),
                    "gamma_winner": winner,
                    "yes_settles_1": yes_1,
                    "global_best": global_best,
                    "best_per_margin": best_per_margin,
                    "grid_points": len(grid),
                    # Back-compat for older dashboard readers
                    "best": global_best,
                }
                f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
                n += 1
    return n
