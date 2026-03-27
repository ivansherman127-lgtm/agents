#!/usr/bin/env python3
"""
Periodically run hedge-cancel sweeps (NO buy range × spread 1–5¢ on YES and NO) over recorded CLOB snapshots.

  PYTHONPATH=. python scripts/python/run_continuous_backtests.py \\
    --interval 900 --out-dir data/backtest_logs --margin-cents-min 1 --margin-cents-max 5 \\
    --n-cycles 1 --cycle-step-cents 0

Optional **--n-cycles** 1–3 repeats a full hedge episode on the remaining book within each window;
**--cycle-step-cents** 0–5 adds that many cents to the spread on each later cycle.

Use the Streamlit dashboard tab **Backtest logs** to inspect ``margin_pct`` and sweep rows.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.application.continuous_backtest import run_hedge_cancel_sweep_log_cycle

DEFAULT_CLOB = REPO_ROOT / "data" / "clob_snapshots"
DEFAULT_OUT = REPO_ROOT / "data" / "backtest_logs"


def main() -> None:
    p = argparse.ArgumentParser(description="Continuous hedge-cancel sweep logging.")
    p.add_argument("--clob-dir", type=Path, default=DEFAULT_CLOB)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--interval", type=float, default=600.0, help="Seconds between cycles")
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-settlement", action="store_true", help="Trade cash only in sim")
    p.add_argument(
        "--margin-cents-min",
        type=int,
        default=1,
        help="Spread lower bound in cents (YES and NO: sell − buy)",
    )
    p.add_argument(
        "--margin-cents-max",
        type=int,
        default=5,
        help="Spread upper bound in cents (YES and NO: sell − buy)",
    )
    p.add_argument("--yes-buy-1", type=float, default=0.45, help="YES round-1 buy limit")
    p.add_argument("--yes-buy-2", type=float, default=0.45, help="YES round-2 buy (if pivot to YES2)")
    p.add_argument("--no-buy-2", type=float, default=0.45, help="NO round-2 buy (if pivot to NO2)")
    p.add_argument(
        "--n-cycles",
        type=int,
        default=1,
        choices=(1, 2, 3, 4, 5),
        help="Hedge episodes per window (after each episode ends, continue on remaining book)",
    )
    p.add_argument(
        "--cycle-step-cents",
        type=int,
        default=0,
        choices=(0, 1, 2, 3, 4, 5),
        help="Extra cents added to spread on each subsequent cycle (0 = same spread every cycle)",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = args.out_dir / f"continuous_{day}.jsonl"
        try:
            n = run_hedge_cancel_sweep_log_cycle(
                args.clob_dir.expanduser(),
                out,
                apply_settlement=not args.no_settlement,
                margin_cents_min=args.margin_cents_min,
                margin_cents_max=args.margin_cents_max,
                yes_buy_1=args.yes_buy_1,
                yes_buy_2=args.yes_buy_2,
                no_buy_2=args.no_buy_2,
                n_cycles=args.n_cycles,
                cycle_step_cents=args.cycle_step_cents,
                fill_policy="limit",
            )
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] wrote {n} record(s) -> {out}",
                flush=True,
            )
        except Exception as e:
            print(f"[error] {e}", flush=True)
        if args.once:
            break
        time.sleep(max(30.0, args.interval))


if __name__ == "__main__":
    main()
