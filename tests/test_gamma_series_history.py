import unittest
from datetime import datetime, timezone

from agents.application.gamma_series_history import (
    event_end_datetime,
    resolved_winner_outcome_label,
    sort_series_events_newest_first,
    window_ts_from_slug,
)


class TestGammaSeriesHistory(unittest.TestCase):
    def test_window_ts_from_slug(self) -> None:
        self.assertEqual(
            window_ts_from_slug("eth-updown-15m-1774373400"), 1774373400
        )
        self.assertEqual(
            window_ts_from_slug("eth-up-or-down-15m-1757724300"), 1757724300
        )
        self.assertIsNone(window_ts_from_slug("no-suffix"))
        self.assertIsNone(window_ts_from_slug(""))

    def test_resolved_up(self) -> None:
        m = {
            "closed": True,
            "outcomes": '["Up", "Down"]',
            "outcomePrices": '["1", "0"]',
        }
        self.assertEqual(resolved_winner_outcome_label(m), "Up")

    def test_resolved_down(self) -> None:
        m = {
            "closed": True,
            "outcomes": ["Up", "Down"],
            "outcomePrices": ["0", "1"],
        }
        self.assertEqual(resolved_winner_outcome_label(m), "Down")

    def test_open_market_no_winner(self) -> None:
        m = {
            "closed": False,
            "outcomes": ["Up", "Down"],
            "outcomePrices": ["0.5", "0.5"],
        }
        self.assertIsNone(resolved_winner_outcome_label(m))

    def test_settled_prices_before_closed_flag(self) -> None:
        """Gamma often leaves closed=false while the UI already shows ~100% / 0%."""
        m = {
            "closed": False,
            "outcomes": ["Up", "Down"],
            "outcomePrices": ["0.9985", "0.0015"],
        }
        self.assertEqual(resolved_winner_outcome_label(m), "Up")

    def test_settled_down_before_closed(self) -> None:
        m = {
            "closed": False,
            "outcomes": ["Up", "Down"],
            "outcomePrices": ["0.02", "0.98"],
        }
        self.assertEqual(resolved_winner_outcome_label(m), "Down")

    def test_event_end_datetime_and_sort(self) -> None:
        a = {"markets": [{"endDate": "2025-09-14T12:00:00Z"}]}
        b = {"markets": [{"endDate": "2026-03-24T18:00:00Z"}]}
        c = {"markets": [{"endDate": "2026-01-01T00:00:00Z"}]}
        evs = [a, b, c]
        sort_series_events_newest_first(evs)
        self.assertEqual(event_end_datetime(evs[0]), datetime(2026, 3, 24, 18, 0, tzinfo=timezone.utc))
        self.assertEqual(event_end_datetime(evs[1]), datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(event_end_datetime(evs[2]), datetime(2025, 9, 14, 12, 0, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
