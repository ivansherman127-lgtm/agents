"""
Symmetric hedge-cancel: place limit buys on **both** YES and NO. **As soon as either round-1
buy fills**, the other leg’s working buy is **cancelled** if it has not filled yet (same snapshot:
YES is evaluated before NO, so if both quotes would fill, YES fills and NO is cancelled).

If the other leg **already** bought before this rule fires, both legs keep working their sells.

After one leg completes its round-1 sell and the other never had inventory, open **round 2** on
the same side that completed round 1 (``yes2`` / ``no2``) at ``*_buy_2`` / ``*_buy_2 + spread``.

The **same** spread (margin in dollars) applies to **both** options in round 1 and round 2:
``sell = buy + spread`` for each leg.

**Multi-cycle (same market window):** after an episode completes, the sim can restart on the
remaining snapshots (up to 5 cycles). **Cycle step** (0–5¢) adds that much to the spread on each
later cycle, increasing the priced-in edge per round-trip versus the previous cycle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.application.clob_snapshot_backtest import best_bid_ask


def _hedge_episode_terminal(
    phase: str,
    sold_y2: bool,
    sold_n2: bool,
) -> bool:
    """Round 1 fully closed both legs, or round 2 on the pivoted side completed a sell."""
    if phase == "done":
        return True
    if phase == "yes2" and sold_y2:
        return True
    if phase == "no2" and sold_n2:
        return True
    return False


def simulate_symmetric_hedge_cancel_segment(
    snapshots: List[Dict[str, Any]],
    start_index: int,
    *,
    yes_buy_1: float,
    no_buy: float,
    spread: float,
    yes_buy_2: float,
    no_buy_2: float,
    size: float,
    yes_expires_at_1: bool,
    apply_settlement: bool,
    stop_at_hedge_terminal: bool,
    fill_policy: str = "limit",
) -> Tuple[Dict[str, Any], int]:
    """
    Run the symmetric hedge state machine from ``snapshots[start_index:]``.

    If ``stop_at_hedge_terminal``, stop after the first row where the episode completes
    (both round-1 legs flat, or round-2 sell filled on the pivoted side) and return
    ``next_index`` = row after that bar. Otherwise consume through the last snapshot.
    """
    yes_sell_1 = round(min(0.99, yes_buy_1 + spread), 2)
    no_sell_1 = round(min(0.99, no_buy + spread), 2)
    yes_sell_2 = round(min(0.99, yes_buy_2 + spread), 2)
    no_sell_2 = round(min(0.99, no_buy_2 + spread), 2)

    phase = "r1"
    yes_buy_cancelled = False
    no_buy_cancelled = False
    pos_y1 = pos_n1 = 0.0
    bought_y1 = bought_n1 = False
    sold_y1 = sold_n1 = False
    pos_y2 = pos_n2 = 0.0
    bought_y2 = sold_y2 = False
    bought_n2 = sold_n2 = False
    cash = 0.0
    yes_buy_qty = yes_buy_notional = 0.0
    no_buy_qty = no_buy_notional = 0.0
    yes_sell_qty = yes_sell_notional = 0.0
    no_sell_qty = no_sell_notional = 0.0
    series: List[Dict[str, Any]] = []
    next_index = len(snapshots)

    n = len(snapshots)
    use_limit_fills = (fill_policy or "limit").lower() == "limit"
    for row_i in range(max(0, start_index), n):
        row = snapshots[row_i]
        yb, ya = best_bid_ask(row.get("yes_book"))
        nb, na = best_bid_ask(row.get("no_book"))

        if phase == "r1":
            if not yes_buy_cancelled and ya is not None and not bought_y1 and ya <= yes_buy_1:
                fill_p = yes_buy_1 if use_limit_fills else ya
                cash -= fill_p * size
                pos_y1 += size
                bought_y1 = True
                yes_buy_qty += size
                yes_buy_notional += fill_p * size
            if not no_buy_cancelled and na is not None and not bought_n1 and na <= no_buy:
                fill_p = no_buy if use_limit_fills else na
                cash -= fill_p * size
                pos_n1 += size
                bought_n1 = True
                no_buy_qty += size
                no_buy_notional += fill_p * size
            # Cancel the other leg's open buy as soon as one side has filled (R1).
            if bought_y1 and not bought_n1 and not no_buy_cancelled:
                no_buy_cancelled = True
            if bought_n1 and not bought_y1 and not yes_buy_cancelled:
                yes_buy_cancelled = True
            if pos_y1 > 0 and not sold_y1 and yb is not None and yb >= yes_sell_1:
                fill_qty = pos_y1
                fill_p = yes_sell_1 if use_limit_fills else yb
                cash += fill_p * pos_y1
                pos_y1 = 0.0
                sold_y1 = True
                yes_sell_qty += fill_qty
                yes_sell_notional += fill_p * fill_qty
            if pos_n1 > 0 and not sold_n1 and nb is not None and nb >= no_sell_1:
                fill_qty = pos_n1
                fill_p = no_sell_1 if use_limit_fills else nb
                cash += fill_p * pos_n1
                pos_n1 = 0.0
                sold_n1 = True
                no_sell_qty += fill_qty
                no_sell_notional += fill_p * fill_qty

            if sold_y1 and sold_n1:
                phase = "done"
            elif sold_y1 and not bought_n1:
                no_buy_cancelled = True
                phase = "yes2"
            elif sold_n1 and not bought_y1:
                yes_buy_cancelled = True
                phase = "no2"

        if phase == "yes2":
            if ya is not None and not bought_y2 and ya <= yes_buy_2:
                fill_p = yes_buy_2 if use_limit_fills else ya
                cash -= fill_p * size
                pos_y2 += size
                bought_y2 = True
                yes_buy_qty += size
                yes_buy_notional += fill_p * size
            if pos_y2 > 0 and not sold_y2 and yb is not None and yb >= yes_sell_2:
                fill_qty = pos_y2
                fill_p = yes_sell_2 if use_limit_fills else yb
                cash += fill_p * pos_y2
                pos_y2 = 0.0
                sold_y2 = True
                yes_sell_qty += fill_qty
                yes_sell_notional += fill_p * fill_qty

        if phase == "no2":
            if na is not None and not bought_n2 and na <= no_buy_2:
                fill_p = no_buy_2 if use_limit_fills else na
                cash -= fill_p * size
                pos_n2 += size
                bought_n2 = True
                no_buy_qty += size
                no_buy_notional += fill_p * size
            if pos_n2 > 0 and not sold_n2 and nb is not None and nb >= no_sell_2:
                fill_qty = pos_n2
                fill_p = no_sell_2 if use_limit_fills else nb
                cash += fill_p * pos_n2
                pos_n2 = 0.0
                sold_n2 = True
                no_sell_qty += fill_qty
                no_sell_notional += fill_p * fill_qty

        series.append(
            {
                "ts_utc": str(row.get("ts_utc") or ""),
                "yes_bid": yb,
                "yes_ask": ya,
                "no_bid": nb,
                "no_ask": na,
                "phase": phase,
                "yes_buy_cancelled": yes_buy_cancelled,
                "no_buy_cancelled": no_buy_cancelled,
            }
        )

        if stop_at_hedge_terminal and _hedge_episode_terminal(phase, sold_y2, sold_n2):
            next_index = row_i + 1
            break

    cash_after_trades = cash
    if apply_settlement:
        if pos_y1 > 0:
            cash += (1.0 if yes_expires_at_1 else 0.0) * pos_y1
        if pos_n1 > 0:
            cash += (0.0 if yes_expires_at_1 else 1.0) * pos_n1
        if pos_y2 > 0:
            cash += (1.0 if yes_expires_at_1 else 0.0) * pos_y2
        if pos_n2 > 0:
            cash += (0.0 if yes_expires_at_1 else 1.0) * pos_n2

    deployed = size * (yes_buy_1 + no_buy)
    if no_buy_cancelled:
        deployed += size * yes_buy_2
    if yes_buy_cancelled:
        deployed += size * no_buy_2

    out = {
        "cash_after_trades": round(cash_after_trades, 6),
        "cash_after_settlement": round(cash, 6),
        "deployed_capital_usdc": round(deployed, 6),
        "return_on_capital_pct": round(
            (100.0 * cash / deployed) if deployed > 0 else 0.0,
            4,
        ),
        "margin_pct": round(
            (100.0 * cash / deployed) if deployed > 0 else 0.0,
            4,
        ),
        "phase": phase,
        "yes_buy_cancelled": yes_buy_cancelled,
        "no_buy_cancelled": no_buy_cancelled,
        "sold_y1": sold_y1,
        "sold_n1": sold_n1,
        "sold_y2": sold_y2,
        "sold_n2": sold_n2,
        "bought_y1": bought_y1,
        "bought_n1": bought_n1,
        "bought_y2": bought_y2,
        "bought_n2": bought_n2,
        "yes_buy_qty": round(yes_buy_qty, 6),
        "yes_buy_notional": round(yes_buy_notional, 6),
        "no_buy_qty": round(no_buy_qty, 6),
        "no_buy_notional": round(no_buy_notional, 6),
        "yes_sell_qty": round(yes_sell_qty, 6),
        "yes_sell_notional": round(yes_sell_notional, 6),
        "no_sell_qty": round(no_sell_qty, 6),
        "no_sell_notional": round(no_sell_notional, 6),
        "spread": spread,
        "series": series,
    }
    return out, next_index


def simulate_symmetric_hedge_cancel_multi_cycle(
    snapshots: List[Dict[str, Any]],
    *,
    yes_buy_1: float,
    no_buy: float,
    base_spread: float,
    yes_buy_2: float,
    no_buy_2: float,
    size: float,
    yes_expires_at_1: bool,
    apply_settlement: bool,
    n_cycles: int,
    cycle_step_cents: int,
    fill_policy: str = "limit",
) -> Dict[str, Any]:
    """
    Run up to ``n_cycles`` hedge episodes on the same window, advancing the snapshot cursor
    after each episode completes.

    ``cycle_step_cents`` (0–5): each subsequent cycle adds this many cents to the **spread**
    (sell = buy + spread), increasing the priced-in edge per round-trip vs the prior cycle.
    """
    n_cycles = max(1, min(5, int(n_cycles)))
    cycle_step_cents = max(0, min(5, int(cycle_step_cents)))
    step = round(cycle_step_cents / 100.0, 2)

    idx = 0
    total_pnl = 0.0
    total_dep = 0.0
    per_cycle: List[Dict[str, Any]] = []
    combined_series: List[Dict[str, Any]] = []

    for c in range(n_cycles):
        if idx >= len(snapshots):
            break
        spread_c = round(base_spread + c * step, 2)
        seg, next_idx = simulate_symmetric_hedge_cancel_segment(
            snapshots,
            idx,
            yes_buy_1=yes_buy_1,
            no_buy=no_buy,
            spread=spread_c,
            yes_buy_2=yes_buy_2,
            no_buy_2=no_buy_2,
            size=size,
            yes_expires_at_1=yes_expires_at_1,
            apply_settlement=apply_settlement,
            stop_at_hedge_terminal=True,
            fill_policy=fill_policy,
        )
        pnl = float(
            seg["cash_after_settlement"]
            if apply_settlement
            else seg["cash_after_trades"]
        )
        dep = float(seg["deployed_capital_usdc"])
        total_pnl += pnl
        total_dep += dep
        row = dict(seg)
        row["cycle_index"] = c
        row["spread_used"] = spread_c
        per_cycle.append(row)
        for pt in seg.get("series") or []:
            combined_series.append({**pt, "cycle_index": c})
        idx = next_idx

    margin_pct = round((100.0 * total_pnl / total_dep) if total_dep > 0 else 0.0, 4)
    yes_buy_qty = sum(float(x.get("yes_buy_qty") or 0.0) for x in per_cycle)
    yes_buy_notional = sum(float(x.get("yes_buy_notional") or 0.0) for x in per_cycle)
    no_buy_qty = sum(float(x.get("no_buy_qty") or 0.0) for x in per_cycle)
    no_buy_notional = sum(float(x.get("no_buy_notional") or 0.0) for x in per_cycle)
    yes_sell_qty = sum(float(x.get("yes_sell_qty") or 0.0) for x in per_cycle)
    yes_sell_notional = sum(float(x.get("yes_sell_notional") or 0.0) for x in per_cycle)
    no_sell_qty = sum(float(x.get("no_sell_qty") or 0.0) for x in per_cycle)
    no_sell_notional = sum(float(x.get("no_sell_notional") or 0.0) for x in per_cycle)
    return {
        "cash_after_trades": round(sum(float(x["cash_after_trades"]) for x in per_cycle), 6)
        if per_cycle
        else 0.0,
        "cash_after_settlement": round(total_pnl, 6),
        "deployed_capital_usdc": round(total_dep, 6),
        "return_on_capital_pct": margin_pct,
        "margin_pct": margin_pct,
        "phase": per_cycle[-1]["phase"] if per_cycle else "r1",
        "yes_buy_cancelled": per_cycle[-1].get("yes_buy_cancelled") if per_cycle else False,
        "no_buy_cancelled": per_cycle[-1].get("no_buy_cancelled") if per_cycle else False,
        "sold_y1": per_cycle[-1].get("sold_y1") if per_cycle else False,
        "sold_n1": per_cycle[-1].get("sold_n1") if per_cycle else False,
        "sold_y2": per_cycle[-1].get("sold_y2") if per_cycle else False,
        "sold_n2": per_cycle[-1].get("sold_n2") if per_cycle else False,
        "yes_buy_qty": round(yes_buy_qty, 6),
        "yes_buy_notional": round(yes_buy_notional, 6),
        "no_buy_qty": round(no_buy_qty, 6),
        "no_buy_notional": round(no_buy_notional, 6),
        "yes_sell_qty": round(yes_sell_qty, 6),
        "yes_sell_notional": round(yes_sell_notional, 6),
        "no_sell_qty": round(no_sell_qty, 6),
        "no_sell_notional": round(no_sell_notional, 6),
        "spread": base_spread,
        "series": combined_series,
        "per_cycle": per_cycle,
        "n_cycles_run": len(per_cycle),
        "n_cycles_requested": n_cycles,
        "cycle_step_cents": cycle_step_cents,
    }


def simulate_symmetric_hedge_cancel_book_snapshots(
    snapshots: List[Dict[str, Any]],
    *,
    yes_buy_1: float,
    no_buy: float,
    spread: float,
    yes_buy_2: float,
    no_buy_2: float,
    size: float,
    yes_expires_at_1: bool,
    apply_settlement: bool = True,
    n_cycles: int = 1,
    cycle_step_cents: int = 0,
    fill_policy: str = "limit",
) -> Dict[str, Any]:
    """
    Book-touch simulation; **either** YES or NO may complete round 1 first.

    With ``n_cycles`` > 1 and/or ``cycle_step_cents`` > 0, runs :func:`simulate_symmetric_hedge_cancel_multi_cycle`.
    Default single pass through all snapshots matches the original one-episode behavior.
    """
    if n_cycles <= 1 and cycle_step_cents == 0:
        seg, _ = simulate_symmetric_hedge_cancel_segment(
            snapshots,
            0,
            yes_buy_1=yes_buy_1,
            no_buy=no_buy,
            spread=spread,
            yes_buy_2=yes_buy_2,
            no_buy_2=no_buy_2,
            size=size,
            yes_expires_at_1=yes_expires_at_1,
            apply_settlement=apply_settlement,
            stop_at_hedge_terminal=False,
            fill_policy=fill_policy,
        )
        return seg
    return simulate_symmetric_hedge_cancel_multi_cycle(
        snapshots,
        yes_buy_1=yes_buy_1,
        no_buy=no_buy,
        base_spread=spread,
        yes_buy_2=yes_buy_2,
        no_buy_2=no_buy_2,
        size=size,
        yes_expires_at_1=yes_expires_at_1,
        apply_settlement=apply_settlement,
        n_cycles=n_cycles,
        cycle_step_cents=cycle_step_cents,
        fill_policy=fill_policy,
    )


def simulate_hedge_cancel_book_snapshots(
    snapshots: List[Dict[str, Any]],
    *,
    yes_buy_1: float,
    yes_sell_1: float,
    no_buy: float,
    no_sell: float,
    yes_buy_2: float,
    yes_sell_2: float,
    size: float,
    yes_expires_at_1: bool,
    apply_settlement: bool = True,
) -> Dict[str, Any]:
    """
    Back-compat: ``spread`` from YES round 1; same spread on both legs; round-2 uses ``yes_buy_2``
    and ``no_buy_2 = yes_buy_2`` (legacy single pivot YES2 price).
    """
    spread = round(yes_sell_1 - yes_buy_1, 2)
    return simulate_symmetric_hedge_cancel_book_snapshots(
        snapshots,
        yes_buy_1=yes_buy_1,
        no_buy=no_buy,
        spread=spread,
        yes_buy_2=yes_buy_2,
        no_buy_2=yes_buy_2,
        size=size,
        yes_expires_at_1=yes_expires_at_1,
        apply_settlement=apply_settlement,
    )


def sweep_no_pair_hedge_cancel(
    snapshots: List[Dict[str, Any]],
    *,
    yes_buy_1: float,
    yes_sell_1: float,
    yes_buy_2: float,
    yes_sell_2: float,
    no_buy_min: float,
    no_buy_max: float,
    spread: float,
    size: float,
    yes_expires_at_1: bool,
    apply_settlement: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Legacy: fixed spread; YES sell derived from ``yes_sell_1``; symmetric sim."""
    results: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_pnl: Optional[float] = None
    n = int(round((no_buy_max - no_buy_min) / 0.01)) + 1
    for i in range(n):
        no_b = round(no_buy_min + 0.01 * i, 2)
        sim = simulate_symmetric_hedge_cancel_book_snapshots(
            snapshots,
            yes_buy_1=yes_buy_1,
            no_buy=no_b,
            spread=spread,
            yes_buy_2=yes_buy_2,
            no_buy_2=yes_buy_2,
            size=size,
            yes_expires_at_1=yes_expires_at_1,
            apply_settlement=apply_settlement,
        )
        pnl = float(
            sim["cash_after_settlement"]
            if apply_settlement
            else sim["cash_after_trades"]
        )
        no_s = round(no_b + spread, 2)
        slim = {
            "no_buy": no_b,
            "no_sell": no_s,
            "total_pnl_usdc": round(pnl, 6),
            "margin_pct": sim["margin_pct"],
            "deployed_usdc": sim["deployed_capital_usdc"],
            "no_buy_cancelled": sim["no_buy_cancelled"],
            "yes_buy_cancelled": sim["yes_buy_cancelled"],
        }
        results.append(slim)
        if best_pnl is None or pnl > best_pnl:
            best_pnl = pnl
            best_row = {
                "best_no_buy": no_b,
                "best_no_sell": no_s,
                "total_pnl_usdc": round(pnl, 6),
                "margin_pct": sim["margin_pct"],
                "deployed_usdc": sim["deployed_capital_usdc"],
                "no_cancelled": sim["no_buy_cancelled"],
            }

    results.sort(key=lambda r: r["no_buy"])
    return results, best_row


def sweep_hedge_cancel_margin_sliding(
    snapshots: List[Dict[str, Any]],
    *,
    yes_buy_1: float,
    yes_buy_2: float,
    no_buy_2: float,
    no_buy_min: float,
    no_buy_max: float,
    margin_cents_min: int = 1,
    margin_cents_max: int = 5,
    size: float,
    yes_expires_at_1: bool,
    apply_settlement: bool,
    n_cycles: int = 1,
    cycle_step_cents: int = 0,
    fill_policy: str = "limit",
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each whole-cent **spread** (1–5¢): **both** YES and NO use ``sell = buy + spread`` in
    round 1; round-2 legs use the same spread on ``yes_buy_2`` / ``no_buy_2``.

    Sweeps ``no_buy`` over ``[no_buy_min, no_buy_max]``.

    ``n_cycles`` (1–5): repeat a full hedge episode on the remaining book after each episode
    ends. ``cycle_step_cents`` (0–5): add this many cents to the spread on each subsequent cycle.
    """
    grid: List[Dict[str, Any]] = []
    global_best: Optional[Dict[str, Any]] = None
    best_pnl_global: Optional[float] = None
    best_per_margin: List[Dict[str, Any]] = []

    for mc in range(margin_cents_min, margin_cents_max + 1):
        spread = round(mc / 100.0, 2)
        best_col: Optional[Dict[str, Any]] = None
        best_pnl_col: Optional[float] = None
        n = int(round((no_buy_max - no_buy_min) / 0.01)) + 1
        for i in range(n):
            no_b = round(no_buy_min + 0.01 * i, 2)
            sim = simulate_symmetric_hedge_cancel_book_snapshots(
                snapshots,
                yes_buy_1=yes_buy_1,
                no_buy=no_b,
                spread=spread,
                yes_buy_2=yes_buy_2,
                no_buy_2=no_buy_2,
                size=size,
                yes_expires_at_1=yes_expires_at_1,
                apply_settlement=apply_settlement,
                n_cycles=n_cycles,
                cycle_step_cents=cycle_step_cents,
                fill_policy=fill_policy,
            )
            pnl = float(
                sim["cash_after_settlement"]
                if apply_settlement
                else sim["cash_after_trades"]
            )
            yes_s1 = round(min(0.99, yes_buy_1 + spread), 2)
            no_s = round(min(0.99, no_b + spread), 2)
            slim = {
                "margin_cents": mc,
                "spread": spread,
                "yes_buy": yes_buy_1,
                "yes_sell": yes_s1,
                "no_buy": no_b,
                "no_sell": no_s,
                "total_pnl_usdc": round(pnl, 6),
                "margin_pct": sim["margin_pct"],
                "deployed_usdc": sim["deployed_capital_usdc"],
                "no_buy_cancelled": sim["no_buy_cancelled"],
                "yes_buy_cancelled": sim["yes_buy_cancelled"],
                "n_cycles": n_cycles,
                "cycle_step_cents": cycle_step_cents,
                "n_cycles_run": sim.get("n_cycles_run", n_cycles),
            }
            grid.append(slim)
            if best_pnl_col is None or pnl > best_pnl_col:
                best_pnl_col = pnl
                best_col = dict(slim)
            if best_pnl_global is None or pnl > best_pnl_global:
                best_pnl_global = pnl
                global_best = dict(slim)

        if best_col is not None:
            best_per_margin.append(best_col)

    return grid, global_best, best_per_margin


def backtest_hedge_chain_reinvest(
    rows_all: List[Dict[str, Any]],
    idx: Dict[str, Any],
    chains: Dict[str, List[str]],
    family_keys: List[str],
    *,
    yes_buy_1: float,
    no_buy_min: float,
    no_buy_max: float,
    margin_cents_min: int,
    margin_cents_max: int,
    yes_buy_2: float,
    no_buy_2: float,
    base_size: float,
    reinvest: bool,
    settle_each_window: Any,
    n_cycles: int = 1,
    cycle_step_cents: int = 0,
    fill_policy: str = "limit",
) -> Dict[str, Any]:
    """
    Run :func:`sweep_hedge_cancel_margin_sliding` per window; optionally compound **size** by
    each window’s realized return (``1 + pnl/deployed``) before the next.
    """
    from agents.application.clob_snapshot_backtest import rows_for_condition

    per_window: List[Dict[str, Any]] = []
    size = float(base_size)
    total_pnl = 0.0
    total_dep = 0.0

    for fk in family_keys:
        for cid in chains.get(fk) or []:
            info = idx[cid]
            snaps = rows_for_condition(rows_all, cid)
            if len(snaps) < 1:
                continue
            yes_1, apply_set = settle_each_window(info)
            grid, gb, bpm = sweep_hedge_cancel_margin_sliding(
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
                apply_settlement=apply_set,
                n_cycles=n_cycles,
                cycle_step_cents=cycle_step_cents,
                fill_policy=fill_policy,
            )
            if not gb:
                continue
            pnl = float(gb.get("total_pnl_usdc") or 0)
            dep = float(gb.get("deployed_usdc") or 0)
            total_pnl += pnl
            total_dep += dep
            per_window.append(
                {
                    "family": fk,
                    "market_slug": info.market_slug,
                    "size_used": size,
                    "pnl_usdc": round(pnl, 6),
                    "deployed_usdc": dep,
                    "margin_pct": gb.get("margin_pct"),
                    "global_best": gb,
                    "n_cycles": n_cycles,
                    "cycle_step_cents": cycle_step_cents,
                    "fill_policy": fill_policy,
                }
            )
            if reinvest and dep > 0:
                r = pnl / dep
                size = max(0.01, size * (1.0 + r))

    agg_ret = (100.0 * total_pnl / total_dep) if total_dep > 0 else 0.0
    return {
        "per_window": per_window,
        "total_pnl_usdc": round(total_pnl, 6),
        "total_deployed_usdc": round(total_dep, 6),
        "aggregate_return_pct": round(agg_ret, 4),
        "reinvest": reinvest,
        "final_size": size,
    }
