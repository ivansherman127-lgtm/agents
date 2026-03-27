import json
import os
import tempfile
import unittest
import sys
import types

from agents.application.dry_run_polymarket import DryRunPolymarket
from agents.application.session_metrics import TradeSessionMetrics

# Stub heavy runtime dependencies before importing Trader.
executor_mod = types.ModuleType("agents.application.executor")


class Executor:
    pass


executor_mod.Executor = Executor
sys.modules["agents.application.executor"] = executor_mod

gamma_mod = types.ModuleType("agents.polymarket.gamma")


class GammaMarketClient:
    pass


gamma_mod.GammaMarketClient = GammaMarketClient
sys.modules["agents.polymarket.gamma"] = gamma_mod

polymarket_mod = types.ModuleType("agents.polymarket.polymarket")


class Polymarket:
    pass


polymarket_mod.Polymarket = Polymarket
sys.modules["agents.polymarket.polymarket"] = polymarket_mod

from agents.application.trade import Trader


class FakePolymarket:
    def __init__(self, fills_sequence, market_open_sequence=None):
        self.fills_sequence = fills_sequence
        self.market_open_sequence = market_open_sequence or [True]
        self._fill_idx = {"YES": 0, "NO": 0}
        self._order_to_side = {}
        self._order_counter = 0
        self.buy_orders = []
        self.sell_orders = []

    def execute_order(self, price, size, side, token_id):
        self._order_counter += 1
        order_id = f"{side.lower()}-{self._order_counter}"
        if side == "BUY":
            side_name = "YES" if token_id == "yes-token" else "NO"
            self._order_to_side[order_id] = side_name
            self.buy_orders.append((side_name, price, size, token_id, order_id))
        else:
            side_name = "YES" if token_id == "yes-token" else "NO"
            self.sell_orders.append((side_name, price, size, token_id, order_id))
        return {"orderID": order_id}

    def extract_order_id(self, order_response):
        return order_response["orderID"]

    def get_order_filled_size(self, order_id):
        side_name = self._order_to_side[order_id]
        seq = self.fills_sequence[side_name]
        idx = self._fill_idx[side_name]
        value = seq[min(idx, len(seq) - 1)]
        self._fill_idx[side_name] += 1
        return value

    def is_market_open_by_slug(self, market_slug):
        value = self.market_open_sequence[0]
        if len(self.market_open_sequence) > 1:
            self.market_open_sequence = self.market_open_sequence[1:]
        return value


class TestOneShotStrategy(unittest.TestCase):
    def _make_trader(self, fake_polymarket):
        trader = Trader.__new__(Trader)
        trader.polymarket = fake_polymarket
        return trader

    def test_places_sells_only_after_full_fill(self):
        fake = FakePolymarket(
            fills_sequence={"YES": [2.0, 5.0], "NO": [1.0, 5.0]},
            market_open_sequence=[True, True, True],
        )
        trader = self._make_trader(fake)
        with tempfile.TemporaryDirectory() as td:
            metrics = TradeSessionMetrics(session_id="full-fill-test", sessions_dir=td)
            trader.run_fixed_market_one_shot(
                yes_token_id="yes-token",
                no_token_id="no-token",
                market_slug="test-market",
                poll_interval_seconds=1,
                max_runtime_seconds=10,
                metrics=metrics,
            )

            self.assertEqual(len(fake.buy_orders), 2)
            self.assertEqual(len(fake.sell_orders), 2)
            for side_name, price, size, _, _ in fake.sell_orders:
                self.assertIn(side_name, {"YES", "NO"})
                self.assertEqual(price, 0.50)
                self.assertEqual(size, 5.0)
            buys = [e for e in metrics.events if e["type"] == "placed_buy"]
            self.assertEqual(len(buys), 2)
            self.assertEqual(metrics.outcome, "done")

    def test_partial_fill_does_not_sell_before_timeout(self):
        fake = FakePolymarket(
            fills_sequence={"YES": [1.0, 2.0, 2.5], "NO": [0.0, 0.5, 1.0]},
            market_open_sequence=[True, True, True],
        )
        trader = self._make_trader(fake)
        with tempfile.TemporaryDirectory() as td:
            metrics = TradeSessionMetrics(session_id="partial-timeout", sessions_dir=td)
            trader.run_fixed_market_one_shot(
                yes_token_id="yes-token",
                no_token_id="no-token",
                market_slug="test-market",
                poll_interval_seconds=1,
                max_runtime_seconds=2,
                metrics=metrics,
            )

            self.assertEqual(len(fake.buy_orders), 2)
            self.assertEqual(len(fake.sell_orders), 0)
            self.assertEqual(metrics.outcome, "max_runtime")

    def test_stops_when_market_closes(self):
        fake = FakePolymarket(
            fills_sequence={"YES": [0.0, 5.0], "NO": [0.0, 5.0]},
            market_open_sequence=[False],
        )
        trader = self._make_trader(fake)
        with tempfile.TemporaryDirectory() as td:
            metrics = TradeSessionMetrics(session_id="market-closed", sessions_dir=td)
            trader.run_fixed_market_one_shot(
                yes_token_id="yes-token",
                no_token_id="no-token",
                market_slug="test-market",
                poll_interval_seconds=1,
                max_runtime_seconds=10,
                metrics=metrics,
            )

            self.assertEqual(len(fake.buy_orders), 2)
            self.assertEqual(len(fake.sell_orders), 0)
            self.assertEqual(metrics.outcome, "market_closed")


class TestMultiCycle(unittest.TestCase):
    def test_two_cycles_instant_dry_run(self):
        trader = Trader.__new__(Trader)
        trader.polymarket = DryRunPolymarket(
            fill_mode="instant", delay_polls=2, market_open=True
        )
        with tempfile.TemporaryDirectory() as td:
            metrics = TradeSessionMetrics(session_id="multi-cycle", sessions_dir=td)
            trader.run_fixed_market_multi_cycle(
                yes_token_id="y",
                no_token_id="n",
                market_slug="",
                buy_price=0.45,
                sell_price=0.50,
                buy_size=5.0,
                poll_interval_seconds=1,
                max_runtime_seconds=60,
                metrics=metrics,
                cycles=2,
                market_url="https://example.com/event/foo",
            )
            self.assertEqual(metrics.outcome, "done")
            buys = [e for e in metrics.events if e["type"] == "placed_buy"]
            self.assertEqual(len(buys), 4)


class TestDryRunAndMetrics(unittest.TestCase):
    def test_dry_run_delayed_reaches_done(self):
        trader = Trader.__new__(Trader)
        trader.polymarket = DryRunPolymarket(
            fill_mode="delayed", delay_polls=2, market_open=True
        )
        with tempfile.TemporaryDirectory() as td:
            metrics = TradeSessionMetrics(session_id="dry-delayed", sessions_dir=td)
            trader.run_fixed_market_one_shot(
                yes_token_id="y",
                no_token_id="n",
                market_slug="",
                poll_interval_seconds=1,
                max_runtime_seconds=30,
                metrics=metrics,
            )
            self.assertEqual(metrics.outcome, "done")
            path = metrics._jsonl_path
            assert path is not None
            self.assertTrue(os.path.isfile(path))
            lines = open(path, encoding="utf-8").read().strip().splitlines()
            types_ = [json.loads(line)["type"] for line in lines]
            self.assertGreaterEqual(types_.count("placed_buy"), 2)
            self.assertGreaterEqual(types_.count("placed_sell"), 2)


if __name__ == "__main__":
    unittest.main()
