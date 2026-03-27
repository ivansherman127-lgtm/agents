import unittest

from agents.application.polymarket_url import slug_from_polymarket_url


class TestPolymarketUrl(unittest.TestCase):
    def test_event_path(self):
        self.assertEqual(
            slug_from_polymarket_url(
                "https://polymarket.com/event/will-it-rain-tomorrow"
            ),
            "will-it-rain-tomorrow",
        )

    def test_market_path(self):
        self.assertEqual(
            slug_from_polymarket_url("https://polymarket.com/market/foo-bar"),
            "foo-bar",
        )


if __name__ == "__main__":
    unittest.main()
