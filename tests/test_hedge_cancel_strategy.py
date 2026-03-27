"""Hedge-cancel book simulation and NO-pair sweep."""

from agents.application.hedge_cancel_strategy import (
    simulate_hedge_cancel_book_snapshots,
    simulate_symmetric_hedge_cancel_book_snapshots,
    sweep_hedge_cancel_margin_sliding,
    sweep_no_pair_hedge_cancel,
)


def test_hedge_cancel_pivots_when_no_never_fills():
    snaps = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "yes_book": {"bids": [], "asks": [{"price": "0.4"}]},
            "no_book": {"bids": [], "asks": [{"price": "0.99"}]},
        },
        {
            "ts_utc": "2026-01-01T00:00:05Z",
            "yes_book": {"bids": [{"price": "0.55"}], "asks": []},
            "no_book": {"bids": [], "asks": [{"price": "0.99"}]},
        },
        {
            "ts_utc": "2026-01-01T00:00:10Z",
            "yes_book": {"bids": [], "asks": [{"price": "0.44"}]},
            "no_book": {"bids": [], "asks": [{"price": "0.99"}]},
        },
        {
            "ts_utc": "2026-01-01T00:00:15Z",
            "yes_book": {"bids": [{"price": "0.52"}], "asks": []},
            "no_book": {"bids": [], "asks": [{"price": "0.99"}]},
        },
    ]
    r = simulate_hedge_cancel_book_snapshots(
        snaps,
        yes_buy_1=0.45,
        yes_sell_1=0.50,
        no_buy=0.45,
        no_sell=0.50,
        yes_buy_2=0.45,
        yes_sell_2=0.50,
        size=10.0,
        yes_expires_at_1=True,
        apply_settlement=True,
    )
    assert r["no_buy_cancelled"] is True
    assert r["sold_y1"] is True
    assert r["sold_y2"] is True


def test_multi_cycle_runs_without_error():
    snaps = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "yes_book": {"asks": [{"price": "0.4"}]},
            "no_book": {"asks": [{"price": "0.5"}]},
        },
    ]
    r = simulate_symmetric_hedge_cancel_book_snapshots(
        snaps,
        yes_buy_1=0.45,
        no_buy=0.45,
        spread=0.05,
        yes_buy_2=0.45,
        no_buy_2=0.45,
        size=5.0,
        yes_expires_at_1=True,
        apply_settlement=True,
        n_cycles=2,
        cycle_step_cents=2,
    )
    assert r.get("n_cycles_run", 0) >= 1
    assert "per_cycle" in r


def test_sliding_margin_sweep_five_columns():
    snaps = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "yes_book": {"asks": [{"price": "0.4"}]},
            "no_book": {"asks": [{"price": "0.5"}]},
        },
    ]
    grid, gb, bpm = sweep_hedge_cancel_margin_sliding(
        snaps,
        yes_buy_1=0.45,
        yes_buy_2=0.45,
        no_buy_2=0.45,
        no_buy_min=0.45,
        no_buy_max=0.46,
        margin_cents_min=1,
        margin_cents_max=5,
        size=5.0,
        yes_expires_at_1=True,
        apply_settlement=True,
    )
    assert len(bpm) == 5
    assert gb is not None
    assert len(grid) == 5 * 2


def test_sweep_returns_rows():
    snaps = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "yes_book": {"asks": [{"price": "0.4"}]},
            "no_book": {"asks": [{"price": "0.5"}]},
        },
    ]
    rows, best = sweep_no_pair_hedge_cancel(
        snaps,
        yes_buy_1=0.45,
        yes_sell_1=0.50,
        yes_buy_2=0.45,
        yes_sell_2=0.50,
        no_buy_min=0.45,
        no_buy_max=0.47,
        spread=0.05,
        size=5.0,
        yes_expires_at_1=True,
        apply_settlement=True,
    )
    assert len(rows) == 3
    assert best is not None
    assert "best_no_buy" in best
