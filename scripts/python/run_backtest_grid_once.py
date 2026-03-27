#!/usr/bin/env python3
"""
Run hedge-cancel logger once for a grid of cycle settings.

This executes one sweep per combination:
- n_cycles in [1, 3]
- cycle_step_cents in [1, 5]

All rows are appended to one daily JSONL file in --out-dir.

Example:
  PYTHONPATH=. python scripts/python/run_backtest_grid_once.py --out-dir data/backtest_logs
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.application.continuous_backtest import run_hedge_cancel_sweep_log_cycle

DEFAULT_CLOB = REPO_ROOT / "data" / "clob_snapshots"
DEFAULT_OUT = REPO_ROOT / "data" / "backtest_logs"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run hedge logger once for every cycles/step combination."
    )
    p.add_argument("--clob-dir", type=Path, default=DEFAULT_CLOB)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-settlement", action="store_true", help="Trade cash only in sim")
    p.add_argument("--margin-cents-min", type=int, default=1)
    p.add_argument("--margin-cents-max", type=int, default=5)
    p.add_argument("--yes-buy-1", type=float, default=0.45)
    p.add_argument("--yes-buy-2", type=float, default=0.45)
    p.add_argument("--no-buy-2", type=float, default=0.45)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = args.out_dir / f"grid_once_{day}.jsonl"

    total_rows = 0
    for n_cycles in range(1, 4):
        for cycle_step_cents in range(1, 6):
            written = run_hedge_cancel_sweep_log_cycle(
                args.clob_dir.expanduser(),
                out,
                apply_settlement=not args.no_settlement,
                margin_cents_min=args.margin_cents_min,
                margin_cents_max=args.margin_cents_max,
                yes_buy_1=args.yes_buy_1,
                yes_buy_2=args.yes_buy_2,
                no_buy_2=args.no_buy_2,
                n_cycles=n_cycles,
                cycle_step_cents=cycle_step_cents,
            )
            total_rows += written
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"cycles={n_cycles} step={cycle_step_cents}c -> {written} row(s)",
                flush=True,
            )

    print(
        f"Completed grid: wrote {total_rows} row(s) to {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
