"""Tests for book-snapshot dual-limit simulation."""

from agents.application.clob_snapshot_backtest import (
    backtest_family_chains,
    best_bid_ask,
    index_family_chains,
    index_recorded_markets,
    market_family_key,
    merged_yes_ask_series_for_chain,
    rows_for_condition,
    simulate_dual_limit_book_snapshots,
)


def test_best_bid_ask_empty():
    assert best_bid_ask(None) == (None, None)
    assert best_bid_ask({}) == (None, None)


def test_best_bid_ask_touch():
    book = {
        "bids": [{"price": "0.4"}, {"price": "0.45"}],
        "asks": [{"price": "0.99"}, {"price": "0.55"}],
    }
    assert best_bid_ask(book) == (0.45, 0.55)


def test_simulate_dual_limit_buy_sell_yes_only():
    snaps = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "yes_book": {"bids": [], "asks": [{"price": "0.4"}]},
            "no_book": {"bids": [], "asks": [{"price": "0.65"}]},
        },
        {
            "ts_utc": "2026-01-01T00:00:10Z",
            "yes_book": {"bids": [{"price": "0.6"}], "asks": []},
            "no_book": {"bids": [], "asks": [{"price": "0.5"}]},
        },
    ]
    r = simulate_dual_limit_book_snapshots(
        snaps,
        buy_price=0.45,
        sell_price=0.55,
        size=10.0,
        yes_expires_at_1=True,
    )
    assert r["bought_yes"] is True
    assert r["sold_yes"] is True
    assert r["bought_no"] is False
    assert r["sold_no"] is False
    assert r["cash_after_settlement"] == round(-0.4 * 10 + 0.6 * 10, 6)


def test_apply_settlement_false_leaves_inventory():
    snaps = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "yes_book": {"bids": [], "asks": [{"price": "0.4"}]},
            "no_book": {"bids": [], "asks": [{"price": "0.65"}]},
        },
    ]
    r = simulate_dual_limit_book_snapshots(
        snaps,
        buy_price=0.45,
        sell_price=0.99,
        size=2.0,
        yes_expires_at_1=True,
        apply_settlement=False,
    )
    assert r["pos_yes_open"] == 2.0
    assert r["pos_no_open"] == 0.0
    assert r["cash_after_trades"] == round(-0.4 * 2, 6)


def test_index_and_rows_for_condition():
    rows = [
        {
            "ts_utc": "2026-01-01T00:00:01Z",
            "condition_id": "0xabc",
            "market_slug": "m1",
            "event_title": "T",
            "bucket": "1H",
            "page_key": "hourly",
            "yes_book": {},
            "no_book": {},
        },
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "condition_id": "0xabc",
            "market_slug": "m1",
            "event_title": "T",
            "bucket": "1H",
            "page_key": "hourly",
            "yes_book": {},
            "no_book": {},
        },
    ]
    idx = index_recorded_markets(rows)
    assert len(idx) == 1
    ordered = rows_for_condition(rows, "0xabc")
    assert [r["ts_utc"] for r in ordered] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
    ]


def test_market_family_key_strips_epoch_suffix():
    assert market_family_key("sol-updown-15m-1774381500") == "sol-updown-15m"
    assert market_family_key("") == ""


def test_merged_series_inserts_break_between_windows():
    rows = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "condition_id": "A",
            "yes_book": {"asks": [{"price": "0.5"}]},
        },
        {
            "ts_utc": "2026-01-01T00:01:00Z",
            "condition_id": "B",
            "yes_book": {"asks": [{"price": "0.6"}]},
        },
    ]
    xs, bid, ask = merged_yes_ask_series_for_chain(rows, ["A", "B"])
    assert xs == [
        "2026-01-01T00:00:00Z",
        None,
        "2026-01-01T00:01:00Z",
    ]
    assert ask == [0.5, None, 0.6]


def test_backtest_chain_resets_each_window():
    """Two windows: each runs its own sim; P/L and deployed sum."""
    rows = [
        {
            "ts_utc": "2026-01-01T00:00:00Z",
            "condition_id": "cA",
            "market_slug": "m-a",
            "event_slug": "x-1000000001",
            "event_title": "X",
            "bucket": "15M",
            "page_key": "15M",
            "yes_book": {"bids": [], "asks": [{"price": "0.4"}]},
            "no_book": {"bids": [], "asks": [{"price": "0.7"}]},
        },
        {
            "ts_utc": "2026-01-01T00:00:10Z",
            "condition_id": "cA",
            "market_slug": "m-a",
            "event_slug": "x-1000000001",
            "event_title": "X",
            "bucket": "15M",
            "page_key": "15M",
            "yes_book": {"bids": [{"price": "0.55"}], "asks": []},
            "no_book": {"bids": [], "asks": [{"price": "0.5"}]},
        },
        {
            "ts_utc": "2026-01-02T00:00:00Z",
            "condition_id": "cB",
            "market_slug": "m-b",
            "event_slug": "x-1000000002",
            "event_title": "X",
            "bucket": "15M",
            "page_key": "15M",
            "yes_book": {"bids": [], "asks": [{"price": "0.4"}]},
            "no_book": {"bids": [], "asks": [{"price": "0.7"}]},
        },
        {
            "ts_utc": "2026-01-02T00:00:10Z",
            "condition_id": "cB",
            "market_slug": "m-b",
            "event_slug": "x-1000000002",
            "event_title": "X",
            "bucket": "15M",
            "page_key": "15M",
            "yes_book": {"bids": [{"price": "0.55"}], "asks": []},
            "no_book": {"bids": [], "asks": [{"price": "0.5"}]},
        },
    ]
    idx = index_recorded_markets(rows)
    chains = index_family_chains(idx)
    assert len(chains) == 1
    fam = next(iter(chains.keys()))
    assert chains[fam] == ["cA", "cB"]

    def settle(_info):
        return True, True

    agg = backtest_family_chains(
        rows,
        idx,
        [fam],
        chains,
        buy_price=0.45,
        sell_price=0.5,
        size=10.0,
        settle_each_window=settle,
    )
    assert agg["n_windows"] == 2
    one_win = -0.4 * 10 + 0.55 * 10
    assert one_win == 1.5
    assert agg["total_pnl_usdc"] == 3.0
    assert agg["total_deployed_usdc"] == round(2 * 0.45 * 10 * 2, 6)
