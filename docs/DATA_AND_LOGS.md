# Data and log files (not in git)

Operational JSONL logs and snapshots live under **`data/`** and are **gitignored** in full. They are not part of the repository; copy them between machines with `rsync` when needed.

| Kind | Typical location |
|------|------------------|
| CLOB snapshots | `data/clob_snapshots/*.jsonl` (monolithic daily files) |
| CLOB by family | `data/clob_snapshots_by_family/*.jsonl` — one file per market family; build with `PYTHONPATH=. python scripts/python/split_clob_snapshots_by_family.py` |
| Benchmark output | `data/clob_snapshots_bench/` |
| Backtest logs | `data/backtest_logs/*.jsonl` |
| Session metrics | `data/sessions/*.jsonl` |
| Root scratch log | `log.txt` (repo root, gitignored) |

**Secrets:** `.env` is gitignored.

If you use the private **`bots`** clone, the same topics are spelled out in that repo’s `docs/COPY_MANIFEST.md`.
