from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from agents.application.executor import Executor as Agent
from agents.polymarket.gamma import GammaMarketClient as Gamma
from agents.polymarket.polymarket import Polymarket

import shutil
import time

if TYPE_CHECKING:
    from agents.application.session_metrics import TradeSessionMetrics


class Trader:
    def __init__(self, polymarket: Optional[Any] = None):
        self.polymarket = polymarket if polymarket is not None else Polymarket()
        self.gamma = Gamma()
        self.agent = Agent()

    def pre_trade_logic(self) -> None:
        self.clear_local_dbs()

    def clear_local_dbs(self) -> None:
        try:
            shutil.rmtree("local_db_events")
        except:
            pass
        try:
            shutil.rmtree("local_db_markets")
        except:
            pass

    def one_best_trade(self) -> None:
        """

        one_best_trade is a strategy that evaluates all events, markets, and orderbooks

        leverages all available information sources accessible to the autonomous agent

        then executes that trade without any human intervention

        """
        try:
            self.pre_trade_logic()

            events = self.polymarket.get_all_tradeable_events()
            print(f"1. FOUND {len(events)} EVENTS")

            filtered_events = self.agent.filter_events_with_rag(events)
            print(f"2. FILTERED {len(filtered_events)} EVENTS")

            markets = self.agent.map_filtered_events_to_markets(filtered_events)
            print()
            print(f"3. FOUND {len(markets)} MARKETS")

            print()
            filtered_markets = self.agent.filter_markets(markets)
            print(f"4. FILTERED {len(filtered_markets)} MARKETS")

            market = filtered_markets[0]
            best_trade = self.agent.source_best_trade(market)
            print(f"5. CALCULATED TRADE {best_trade}")

            amount = self.agent.format_trade_prompt_for_execution(best_trade)
            # Please refer to TOS before uncommenting: polymarket.com/tos
            # trade = self.polymarket.execute_market_order(market, amount)
            # print(f"6. TRADED {trade}")

        except Exception as e:
            print(f"Error {e} \n \n Retrying")
            self.one_best_trade()

    def run_fixed_market_one_shot(
        self,
        yes_token_id: str,
        no_token_id: str,
        market_slug: str = "",
        buy_price: float = 0.45,
        buy_size: float = 5.0,
        sell_price: float = 0.50,
        poll_interval_seconds: int = 3,
        max_runtime_seconds: int = 3600,
        metrics: Optional["TradeSessionMetrics"] = None,
        cycle_index: int = 1,
        total_cycles: int = 1,
        finalize_session: bool = True,
        market_url: str = "",
    ) -> str:
        """
        One-shot strategy (one cycle): limit BUY both outcomes, wait full fill, limit SELL.

        For multiple cycles use :meth:`run_fixed_market_multi_cycle` or call repeatedly with
        ``cycle_index`` / ``finalize_session`` set appropriately.

        Returns outcome string: ``done``, ``max_runtime``, or ``market_closed``.
        """
        from agents.application.session_metrics import TradeSessionMetrics as TSM

        outcomes = {"YES": yes_token_id, "NO": no_token_id}
        state = {
            "buy_submitted": {"YES": False, "NO": False},
            "buy_order_id": {"YES": None, "NO": None},
            "sell_submitted": {"YES": False, "NO": False},
            "sell_order_id": {"YES": None, "NO": None},
        }
        last_reported_fill: dict = {"YES": None, "NO": None}
        session_start = time.monotonic()

        if metrics is None:
            metrics = TSM()
        metrics.market_slug = market_slug or metrics.market_slug

        from agents.application.run_control import (
            clear_session_stop_files,
            stop_pending,
        )

        if stop_pending(metrics.session_id, market_slug):
            print("[STOP] User-requested stop before opening orders.")
            metrics.record("user_stop", phase="before_buys", session_id=metrics.session_id)
            metrics.set_outcome("user_stopped")
            clear_session_stop_files(metrics.session_id, market_slug)
            return "user_stopped"

        if cycle_index <= 1:
            metrics.record(
                "session_start",
                session_id=metrics.session_id,
                market_slug=market_slug,
                market_url=market_url or None,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                buy_price=buy_price,
                buy_size=buy_size,
                sell_price=sell_price,
                max_runtime_seconds=max_runtime_seconds,
                poll_interval_seconds=poll_interval_seconds,
                cycle_index=cycle_index,
                total_cycles=total_cycles,
            )
        else:
            metrics.record(
                "cycle_start",
                session_id=metrics.session_id,
                cycle=cycle_index,
                total_cycles=total_cycles,
                market_slug=market_slug,
                market_url=market_url or None,
                buy_price=buy_price,
                buy_size=buy_size,
                sell_price=sell_price,
            )

        if stop_pending(metrics.session_id, market_slug):
            print("[STOP] User-requested stop after session header (before BUY orders).")
            metrics.record("user_stop", phase="after_header", session_id=metrics.session_id)
            metrics.set_outcome("user_stopped")
            clear_session_stop_files(metrics.session_id, market_slug)
            return "user_stopped"

        for side_name, token_id in outcomes.items():
            response = self.polymarket.execute_order(
                price=buy_price, size=buy_size, side="BUY", token_id=token_id
            )
            order_id = self.polymarket.extract_order_id(response)
            if not order_id:
                metrics.set_outcome("error")
                raise RuntimeError(
                    f"Failed to extract BUY order id for {side_name}. Response: {response}"
                )

            state["buy_submitted"][side_name] = True
            state["buy_order_id"][side_name] = order_id
            metrics.record(
                "placed_buy",
                side=side_name,
                token_id=token_id,
                order_id=order_id,
                size=buy_size,
                price=buy_price,
            )
            print(
                f"[PLACED_BUY] side={side_name} token_id={token_id} order_id={order_id} "
                f"size={buy_size} price={buy_price}"
            )

        start_time = time.time()
        poll_iteration = 0
        while True:
            poll_iteration += 1
            elapsed = time.time() - start_time
            metrics.record(
                "poll_iteration",
                iteration=poll_iteration,
                elapsed_seconds=round(elapsed, 3),
            )

            if stop_pending(metrics.session_id, market_slug):
                print("[STOP] User-requested stop during polling.")
                metrics.record(
                    "user_stop",
                    phase="poll",
                    iteration=poll_iteration,
                    session_id=metrics.session_id,
                )
                metrics.set_outcome("user_stopped")
                clear_session_stop_files(metrics.session_id, market_slug)
                return "user_stopped"

            if elapsed > max_runtime_seconds:
                print("[STOP] Reached max runtime without full completion.")
                metrics.set_outcome("max_runtime")
                return "max_runtime"

            if market_slug:
                market_open = self.polymarket.is_market_open_by_slug(market_slug)
                if not market_open:
                    print("[STOP] Market is closed/resolved/archived. Exiting strategy.")
                    metrics.set_outcome("market_closed")
                    return "market_closed"

            for side_name, token_id in outcomes.items():
                if state["sell_submitted"][side_name]:
                    continue

                buy_order_id = state["buy_order_id"][side_name]
                filled_size = self.polymarket.get_order_filled_size(buy_order_id)
                if last_reported_fill[side_name] != filled_size:
                    metrics.record(
                        "wait_fill",
                        side=side_name,
                        buy_order_id=buy_order_id,
                        filled=filled_size,
                        target=buy_size,
                    )
                    last_reported_fill[side_name] = filled_size
                print(
                    f"[WAIT_FILL] side={side_name} buy_order_id={buy_order_id} "
                    f"filled={filled_size}/{buy_size}"
                )

                if filled_size >= buy_size:
                    sell_response = self.polymarket.execute_order(
                        price=sell_price,
                        size=buy_size,
                        side="SELL",
                        token_id=token_id,
                    )
                    sell_order_id = self.polymarket.extract_order_id(sell_response)
                    if not sell_order_id:
                        metrics.set_outcome("error")
                        raise RuntimeError(
                            f"Failed to extract SELL order id for {side_name}. "
                            f"Response: {sell_response}"
                        )

                    state["sell_submitted"][side_name] = True
                    state["sell_order_id"][side_name] = sell_order_id
                    metrics.record(
                        "placed_sell",
                        side=side_name,
                        token_id=token_id,
                        order_id=sell_order_id,
                        size=buy_size,
                        price=sell_price,
                        buy_price=buy_price,
                    )
                    print(
                        f"[PLACED_SELL] side={side_name} token_id={token_id} "
                        f"order_id={sell_order_id} size={buy_size} price={sell_price}"
                    )

            if all(state["sell_submitted"].values()):
                print("[DONE] One-shot strategy completed. No further orders will be placed.")
                wall_s = time.monotonic() - session_start
                metrics.record(
                    "done",
                    wall_time_seconds=round(wall_s, 4),
                    cycle_index=cycle_index,
                    total_cycles=total_cycles,
                    estimated_pnl_usdc_ex_fees=metrics.summary()[
                        "estimated_pnl_usdc_ex_fees"
                    ],
                )
                if finalize_session:
                    metrics.set_outcome("done")
                return "done"

            time.sleep(max(1, poll_interval_seconds))

    def run_fixed_market_hedge_cancel(
        self,
        yes_token_id: str,
        no_token_id: str,
        market_slug: str = "",
        yes_buy_1: float = 0.45,
        yes_sell_1: float = 0.50,
        no_buy_price: float = 0.45,
        yes_buy_2: float = 0.45,
        yes_sell_2: float = 0.50,
        buy_size: float = 5.0,
        poll_interval_seconds: int = 3,
        max_runtime_seconds: int = 3600,
        metrics: Optional["TradeSessionMetrics"] = None,
        market_url: str = "",
    ) -> str:
        """
        Hedge-cancel: when YES round-1 buy+sell **fully fills**, cancel NO buy if NO has **zero**
        fill; otherwise complete NO as usual. After cancel, place YES round-2 at ``yes_buy_2`` /
        ``yes_sell_2``. Waits for **sell** orders to fill (stricter than plain one-shot).
        """
        from agents.application.session_metrics import TradeSessionMetrics as TSM

        no_sell_price = round(no_buy_price + 0.05, 2)
        if metrics is None:
            metrics = TSM()

        from agents.application.run_control import (
            clear_session_stop_files,
            stop_pending,
        )

        if stop_pending(metrics.session_id, market_slug):
            metrics.set_outcome("user_stopped")
            return "user_stopped"

        metrics.record(
            "session_start",
            session_id=metrics.session_id,
            market_slug=market_slug,
            market_url=market_url or None,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            strategy="hedge_cancel",
            yes_buy_1=yes_buy_1,
            yes_sell_1=yes_sell_1,
            no_buy_price=no_buy_price,
            no_sell_price=no_sell_price,
            yes_buy_2=yes_buy_2,
            yes_sell_2=yes_sell_2,
            buy_size=buy_size,
        )

        yes_buy_oid = no_buy_oid = ""
        yes_sell_oid = no_sell_oid = ""
        yes2_buy_oid = yes2_sell_oid = ""
        no_cancelled = False
        pivot_done = False
        session_start = time.monotonic()
        start_time = time.time()
        poll_iteration = 0

        def _place_buy(token: str, price: float, side_lbl: str) -> str:
            r = self.polymarket.execute_order(
                price=price, size=buy_size, side="BUY", token_id=token
            )
            oid = self.polymarket.extract_order_id(r)
            if not oid:
                raise RuntimeError(f"BUY {side_lbl} failed: {r}")
            metrics.record(
                "placed_buy",
                side=side_lbl,
                token_id=token,
                order_id=oid,
                size=buy_size,
                price=price,
                strategy="hedge_cancel",
            )
            return oid

        def _place_sell(token: str, price: float, side_lbl: str, bp: float) -> str:
            r = self.polymarket.execute_order(
                price=price, size=buy_size, side="SELL", token_id=token
            )
            oid = self.polymarket.extract_order_id(r)
            if not oid:
                raise RuntimeError(f"SELL {side_lbl} failed: {r}")
            metrics.record(
                "placed_sell",
                side=side_lbl,
                token_id=token,
                order_id=oid,
                size=buy_size,
                price=price,
                buy_price=bp,
                strategy="hedge_cancel",
            )
            return oid

        yes_buy_oid = _place_buy(yes_token_id, yes_buy_1, "YES")
        no_buy_oid = _place_buy(no_token_id, no_buy_price, "NO")

        while True:
            poll_iteration += 1
            elapsed = time.time() - start_time
            metrics.record("poll_iteration", iteration=poll_iteration, strategy="hedge_cancel")
            if elapsed > max_runtime_seconds:
                metrics.set_outcome("max_runtime")
                return "max_runtime"
            if stop_pending(metrics.session_id, market_slug):
                metrics.set_outcome("user_stopped")
                clear_session_stop_files(metrics.session_id, market_slug)
                return "user_stopped"
            if market_slug and not self.polymarket.is_market_open_by_slug(market_slug):
                metrics.set_outcome("market_closed")
                return "market_closed"

            ybf = self.polymarket.get_order_filled_size(yes_buy_oid)
            nbf = self.polymarket.get_order_filled_size(no_buy_oid)

            if ybf >= buy_size and not yes_sell_oid:
                yes_sell_oid = _place_sell(yes_token_id, yes_sell_1, "YES", yes_buy_1)
            if not no_cancelled and nbf >= buy_size and not no_sell_oid:
                no_sell_oid = _place_sell(no_token_id, no_sell_price, "NO", no_buy_price)

            yes1_done = False
            if yes_sell_oid:
                ysf = self.polymarket.get_order_filled_size(yes_sell_oid)
                yes1_done = ysf >= buy_size
            no1_done = False
            if no_sell_oid:
                nsf = self.polymarket.get_order_filled_size(no_sell_oid)
                no1_done = nsf >= buy_size

            if (
                yes1_done
                and not pivot_done
                and not no_cancelled
            ):
                pivot_done = True
                if nbf <= 0:
                    self.polymarket.cancel_order(no_buy_oid)
                    metrics.record(
                        "hedge_cancel_no_buy",
                        cancelled_order_id=no_buy_oid,
                        no_buy_filled=nbf,
                    )
                    no_cancelled = True
                    yes2_buy_oid = _place_buy(yes_token_id, yes_buy_2, "YES_R2")
                else:
                    metrics.record(
                        "hedge_maintain_no",
                        no_buy_filled=nbf,
                    )

            if no_cancelled:
                if yes2_buy_oid and not yes2_sell_oid:
                    y2f = self.polymarket.get_order_filled_size(yes2_buy_oid)
                    if y2f >= buy_size:
                        yes2_sell_oid = _place_sell(
                            yes_token_id, yes_sell_2, "YES_R2", yes_buy_2
                        )
                if yes2_sell_oid:
                    y2sf = self.polymarket.get_order_filled_size(yes2_sell_oid)
                    if y2sf >= buy_size:
                        wall_s = time.monotonic() - session_start
                        metrics.record(
                            "done",
                            strategy="hedge_cancel",
                            wall_time_seconds=round(wall_s, 4),
                            pivot="cancel_no",
                            estimated_pnl_usdc_ex_fees=metrics.summary()[
                                "estimated_pnl_usdc_ex_fees"
                            ],
                        )
                        metrics.set_outcome("done")
                        return "done"
            else:
                if yes1_done and no1_done:
                    wall_s = time.monotonic() - session_start
                    metrics.record(
                        "done",
                        strategy="hedge_cancel",
                        wall_time_seconds=round(wall_s, 4),
                        pivot="none",
                        estimated_pnl_usdc_ex_fees=metrics.summary()[
                            "estimated_pnl_usdc_ex_fees"
                        ],
                    )
                    metrics.set_outcome("done")
                    return "done"

            time.sleep(max(1, poll_interval_seconds))

    def run_fixed_market_multi_cycle(
        self,
        yes_token_id: str,
        no_token_id: str,
        market_slug: str = "",
        buy_price: float = 0.45,
        buy_size: float = 5.0,
        sell_price: float = 0.50,
        poll_interval_seconds: int = 3,
        max_runtime_seconds: int = 3600,
        metrics: Optional["TradeSessionMetrics"] = None,
        cycles: int = 1,
        market_url: str = "",
    ) -> str:
        """Run :meth:`run_fixed_market_one_shot` ``cycles`` times in sequence for one market."""
        from agents.application.session_metrics import TradeSessionMetrics as TSM

        n = max(1, cycles)
        if metrics is None:
            metrics = TSM()

        from agents.application.run_control import (
            clear_session_stop_files,
            stop_pending,
        )

        last = "done"
        for c in range(1, n + 1):
            if stop_pending(metrics.session_id, market_slug):
                print("[STOP] User-requested stop between cycles.")
                metrics.record(
                    "user_stop",
                    phase="between_cycles",
                    cycle=c,
                    session_id=metrics.session_id,
                )
                metrics.set_outcome("user_stopped")
                clear_session_stop_files(metrics.session_id, market_slug)
                return "user_stopped"

            last = self.run_fixed_market_one_shot(
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                market_slug=market_slug,
                buy_price=buy_price,
                buy_size=buy_size,
                sell_price=sell_price,
                poll_interval_seconds=poll_interval_seconds,
                max_runtime_seconds=max_runtime_seconds,
                metrics=metrics,
                cycle_index=c,
                total_cycles=n,
                finalize_session=(c == n),
                market_url=market_url,
            )
            if last != "done":
                return last
        return last

    def maintain_positions(self):
        pass

    def incentive_farm(self):
        pass


if __name__ == "__main__":
    t = Trader()
    t.one_best_trade()
