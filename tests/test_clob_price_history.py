import unittest
from unittest.mock import MagicMock, patch

from agents.application.clob_price_history import (
    forward_filled_merge,
    fetch_prices_history,
    iso8601_to_unix,
    parse_clob_token_ids,
    sample_interval_ms_to_fidelity_str,
    simulate_one_shot_dual_limit,
    window_unix_from_gamma_market,
)


class TestClobPriceHistory(unittest.TestCase):
    def test_sample_interval_ms_to_fidelity_str(self) -> None:
        self.assertEqual(sample_interval_ms_to_fidelity_str(60_000), "1")
        self.assertEqual(sample_interval_ms_to_fidelity_str(30_000), "0.5")
        with self.assertRaises(ValueError):
            sample_interval_ms_to_fidelity_str(0)

    def test_iso8601_to_unix(self) -> None:
        self.assertEqual(
            iso8601_to_unix("2026-03-24T17:45:00Z"),
            int(iso8601_to_unix("2026-03-24T17:45:00+00:00")),
        )

    def test_parse_clob_token_ids(self) -> None:
        m = {"clobTokenIds": '["111", "222"]'}
        self.assertEqual(parse_clob_token_ids(m), ("111", "222"))

    def test_window_unix(self) -> None:
        m = {
            "eventStartTime": "2026-03-24T17:30:00Z",
            "endDate": "2026-03-24T17:45:00Z",
        }
        a, b = window_unix_from_gamma_market(m)
        self.assertLess(a, b)

    def test_forward_filled_merge(self) -> None:
        y = [{"t": 100, "p": 0.4}, {"t": 200, "p": 0.6}]
        n = [{"t": 150, "p": 0.6}]
        rows = forward_filled_merge(y, n)
        self.assertEqual(len(rows), 3)

    @patch("agents.application.clob_price_history._http_get_json")
    def test_fetch_prices_history_parses(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "history": [{"t": 10, "p": "0.5"}, {"t": 20, "p": 0.6}],
        }
        h = fetch_prices_history("tok", 1, 100, sample_interval_ms=300_000)
        self.assertEqual(len(h), 2)
        self.assertEqual(h[0]["t"], 10)
        self.assertEqual(h[1]["p"], 0.6)

    def test_simulate_one_shot(self) -> None:
        aligned = [
            {"t": 1, "yes_p": 0.4, "no_p": 0.4},
            {"t": 2, "yes_p": 0.55, "no_p": 0.45},
            {"t": 3, "yes_p": 0.55, "no_p": 0.55},
        ]
        r = simulate_one_shot_dual_limit(
            aligned,
            buy_price=0.45,
            sell_price=0.5,
            size=1.0,
            yes_expires_at_1=True,
        )
        self.assertIn("cash_after_settlement", r)


if __name__ == "__main__":
    unittest.main()
