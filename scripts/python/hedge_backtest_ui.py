"""
Hedge-cancel snapshot backtest UI: parameter sweep, family totals, per-market instance charts.

Run from repo root:

  streamlit run scripts/python/hedge_backtest_ui.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _find_repo_root() -> Path:
    """Directory that contains the ``agents/`` package (parent of ``agents.application``)."""
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "agents" / "application" / "clob_snapshot_backtest.py").is_file():
            return p
    raise RuntimeError(
        "Could not locate repo root (folder containing agents/application/). "
        f"Started from {here}."
    )


REPO_ROOT = _find_repo_root()
_root = str(REPO_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from agents.application.clob_snapshot_backtest import (
        best_bid_ask,
        group_rows_by_condition,
        index_family_chains,
        index_recorded_markets,
        list_split_family_files,
        load_jsonl_snapshots,
        parse_ts_utc,
        rows_for_condition,
        window_outcome_book_extremes,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import agents.application.clob_snapshot_backtest from REPO_ROOT={REPO_ROOT}: {e!r}. "
        "Run `streamlit run` from the repository root (directory that contains the `agents/` package)."
    ) from e
from agents.application.continuous_backtest import _settlement_yes_at_1_for_slug
from agents.application.hedge_cancel_strategy import simulate_symmetric_hedge_cancel_book_snapshots

DEFAULT_CLOB = REPO_ROOT / "data" / "clob_snapshots"
DEFAULT_SPLIT_FAMILIES = REPO_ROOT / "data" / "clob_snapshots_by_family"
DEFAULT_SQLITE_DB = REPO_ROOT / "data" / "clob_snapshots" / "clob_snapshots.db"


def _sqlite_stack():
    """
    Lazy import for SQLite snapshot helpers so JSONL-only runs still start if this stack fails.
    """
    from agents.application.clob_snapshot_backtest import load_all_from_sqlite
    from agents.application.clob_snapshot_store import (
        count_rows_sqlite,
        list_family_keys_sqlite,
        load_rows_from_sqlite,
    )

    return load_all_from_sqlite, count_rows_sqlite, list_family_keys_sqlite, load_rows_from_sqlite


@st.cache_data(ttl=300, show_spinner=False)
def _settlement_cached(market_slug: str) -> Tuple[bool, Optional[str]]:
    return _settlement_yes_at_1_for_slug(market_slug)


def _parse_buy_levels(text: str) -> List[float]:
    """Parse '10,20,30' or range '10-80:10' (cents). Empty -> 10..80 step 10."""
    s = (text or "").strip()
    if not s:
        return [round(i / 100.0, 2) for i in range(10, 81, 10)]
    if "-" in s and ":" in s:
        a, b = s.split(":", 1)
        lo_s, hi_s = a.split("-", 1)
        lo, hi, step = int(lo_s.strip()), int(hi_s.strip()), int(b.strip())
        return [round(x / 100.0, 2) for x in range(lo, hi + 1, step)]
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(round(int(part) / 100.0, 2))
    return sorted(set(out))


def _load_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for p in paths:
        merged.extend(load_jsonl_snapshots(p))
    return merged


def _family_keys_from_rows(rows_all: List[Dict[str, Any]]) -> List[str]:
    idx = index_recorded_markets(rows_all)
    chains = index_family_chains(idx)
    return sorted(chains.keys())


def _family_core_name(family_key: str) -> str:
    """Strip leading ``[15M]`` / ``[1H]`` style prefix for readable dropdown labels."""
    s = (family_key or "").strip()
    m = re.match(r"^\[[^\]]+\]\s*(.+)$", s)
    return (m.group(1).strip() if m else s) or s


def _run_sweep(
    rows_all: List[Dict[str, Any]],
    buy_levels: List[float],
    margin_cents: int,
    n_cycles: int,
    cycle_step_cents: int,
    size: float,
    fill_policy: str,
    apply_settlement: bool,
    *,
    family_keys_filter: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, float], Dict[str, float]]]:
    """
    Returns:
      instance_rows: one dict per (window, buy_level)
      agg_family: (family_key, buy) -> {"pnl":, "dep":}

    If ``family_keys_filter`` is set, only those family keys (intersected with chains) run.
    """
    idx = index_recorded_markets(rows_all)
    chains = index_family_chains(idx)
    family_keys = sorted(chains.keys())
    if family_keys_filter is not None:
        allowed = {k for k in family_keys if k in family_keys_filter}
        if not allowed:
            return [], {}
        family_keys = sorted(allowed)

    cid_yes1: Dict[str, bool] = {}
    for fk in family_keys:
        for cid in chains.get(fk) or []:
            info = idx[cid]
            y1, _ = _settlement_cached(info.market_slug or "")
            cid_yes1[cid] = y1

    spread = round(margin_cents / 100.0, 2)
    instance_rows: List[Dict[str, Any]] = []
    agg_family: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(
        lambda: {"pnl": 0.0, "dep": 0.0}
    )

    for fk in family_keys:
        for cid in chains.get(fk) or []:
            info = idx[cid]
            snaps = rows_for_condition(rows_all, cid)
            if not snaps:
                continue
            y1 = cid_yes1[cid]
            for buy in buy_levels:
                sim = simulate_symmetric_hedge_cancel_book_snapshots(
                    snaps,
                    yes_buy_1=buy,
                    no_buy=buy,
                    spread=spread,
                    yes_buy_2=buy,
                    no_buy_2=buy,
                    size=size,
                    yes_expires_at_1=y1,
                    apply_settlement=apply_settlement,
                    n_cycles=n_cycles,
                    cycle_step_cents=cycle_step_cents,
                    fill_policy=fill_policy,
                )
                cash_key = "cash_after_settlement" if apply_settlement else "cash_after_trades"
                pnl = float(sim.get(cash_key) or 0)
                dep = float(sim.get("deployed_capital_usdc") or 0)
                ret = (100.0 * pnl / dep) if dep > 0 else 0.0
                buy_c = int(round(buy * 100))
                sell_c = int(round(min(0.99, buy + spread) * 100))
                row = {
                    "family": fk,
                    "condition_id": cid,
                    "market_slug": info.market_slug,
                    "bucket": info.bucket,
                    "t_first_iso": info.t_first_iso,
                    "t_last_iso": info.t_last_iso,
                    "buy_cents": buy_c,
                    "sell_cents": sell_c,
                    "pnl_usdc": round(pnl, 6),
                    "deployed_usdc": round(dep, 6),
                    "ret_pct": round(ret, 4),
                    "margin_pct_sim": sim.get("margin_pct"),
                    "yes_cancelled": sim.get("yes_buy_cancelled"),
                    "no_cancelled": sim.get("no_buy_cancelled"),
                }
                instance_rows.append(row)
                key = (fk, buy)
                agg_family[key]["pnl"] += pnl
                agg_family[key]["dep"] += dep

    return instance_rows, dict(agg_family)


def _build_family_tables(
    agg_family: Dict[Tuple[str, float], Dict[str, float]],
    families: List[str],
    buy_levels: List[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return rows for return-% table and PnL table."""
    ret_rows: List[Dict[str, Any]] = []
    pnl_rows: List[Dict[str, Any]] = []
    for fk in families:
        r_row: Dict[str, Any] = {"family": fk, "best_buy_cents": None}
        p_row: Dict[str, Any] = {"family": fk}
        best_r: Optional[float] = None
        best_buy: Optional[int] = None
        for buy in buy_levels:
            pnl = agg_family.get((fk, buy), {"pnl": 0.0, "dep": 0.0})["pnl"]
            dep = agg_family.get((fk, buy), {"pnl": 0.0, "dep": 0.0})["dep"]
            r = (100.0 * pnl / dep) if dep > 0 else 0.0
            c = int(round(buy * 100))
            r_row[f"ret_{c}"] = round(r, 2)
            p_row[f"pnl_{c}"] = round(pnl, 2)
            if best_r is None or r > best_r:
                best_r = r
                best_buy = c
        r_row["best_buy_cents"] = best_buy
        ret_rows.append(r_row)
        pnl_rows.append(p_row)
    return ret_rows, pnl_rows


def _tab_label_family(fk: str, max_len: int = 36) -> str:
    return fk if len(fk) <= max_len else fk[: max_len - 1] + "…"


def _streamlit_key(prefix: str, label: str) -> str:
    """Stable short key for Streamlit widgets (family keys can be long / odd characters)."""
    h = hashlib.md5(label.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def _bid_ask_series_for_snaps(
    snaps: List[Dict[str, Any]],
) -> Tuple[List[float], List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Relative time (s from first snapshot) and best bid/ask for YES and NO books."""
    if not snaps:
        return [], [], [], [], []
    t0 = parse_ts_utc(str(snaps[0].get("ts_utc") or ""))
    x_rel: List[float] = []
    y_bid: List[Optional[float]] = []
    y_ask: List[Optional[float]] = []
    n_bid: List[Optional[float]] = []
    n_ask: List[Optional[float]] = []
    for row in snaps:
        ts = parse_ts_utc(str(row.get("ts_utc") or ""))
        x_rel.append(ts - t0)
        yb, ya = best_bid_ask(row.get("yes_book"))
        nb, na = best_bid_ask(row.get("no_book"))
        y_bid.append(yb)
        y_ask.append(ya)
        n_bid.append(nb)
        n_ask.append(na)
    return x_rel, y_bid, y_ask, n_bid, n_ask


def _render_book_price_ranges_section(rows_all: List[Dict[str, Any]]) -> None:
    """One figure per market family: summary bars + optional full bid/ask curves per window."""
    st.subheader("Book high / low per market window")
    st.caption(
        "Summary: for each **window** (one condition), **low** = minimum best **bid**, **high** = maximum best **ask** "
        "across snapshots. Below, pick a window to plot **full best bid / best ask** over time (YES and NO panels). "
        "One family at a time keeps the page responsive on large datasets."
    )
    # One O(n) pass; avoids O(n×windows) from repeated rows_for_condition scans.
    by_cid = group_rows_by_condition(rows_all)
    idx: Dict[str, Any] = index_recorded_markets(rows_all)
    chains = index_family_chains(idx)
    fks = sorted(chains.keys())
    if not fks:
        st.warning("No market families in loaded snapshots.")
        return

    fk = st.selectbox(
        "Market family (book charts)",
        options=fks,
        format_func=_tab_label_family,
        key="book_chart_family_select",
        help="Charts render only this family (Streamlit runs all tab bodies on every interaction — tabs were slow).",
    )
    _render_family_window_price_figure(by_cid, chains, idx, fk)


_MAX_CURVE_POINTS = 12_000


def _render_window_bid_ask_curves(
    snaps: List[Dict[str, Any]],
    family_title: str,
    window_label: str,
    slug_hint: str,
) -> None:
    """Line chart: best bid & best ask vs time within one window (YES and NO subplots)."""
    n_raw = len(snaps)
    if n_raw > _MAX_CURVE_POINTS:
        step = max(1, n_raw // _MAX_CURVE_POINTS)
        snaps = snaps[::step]
        st.caption(
            f"Plot downsampling: {n_raw:,} snapshots → every {step}th point ({len(snaps):,} points) for speed."
        )
    x_rel, y_bid, y_ask, n_bid, n_ask = _bid_ask_series_for_snaps(snaps)
    if not x_rel:
        st.caption("_No snapshots for this window._")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "YES token — best bid & best ask over time",
            "NO token — best bid & best ask over time",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x_rel,
            y=y_bid,
            mode="lines",
            name="YES bid",
            line=dict(color="#1e8449", width=1.8),
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_rel,
            y=y_ask,
            mode="lines",
            name="YES ask",
            line=dict(color="#7dcea0", width=1.8, dash="dash"),
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_rel,
            y=n_bid,
            mode="lines",
            name="NO bid",
            line=dict(color="#922b21", width=1.8),
            connectgaps=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_rel,
            y=n_ask,
            mode="lines",
            name="NO ask",
            line=dict(color="#e6b0aa", width=1.8, dash="dash"),
            connectgaps=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[0, 1], title="Price", row=1, col=1)
    fig.update_yaxes(range=[0, 1], title="Price", row=2, col=1)
    fig.update_xaxes(title_text="Seconds from first snapshot in window", row=2, col=1)
    n_pts = len(x_rel)
    fig.update_layout(
        title=f"{family_title} — {window_label} ({n_pts} snapshots)",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60),
    )
    st.caption(f"Market: `{slug_hint[:120]}{'…' if len(slug_hint) > 120 else ''}`")
    st.plotly_chart(fig, use_container_width=True)


def _render_family_window_price_figure(
    by_cid: Dict[str, List[Dict[str, Any]]],
    chains: Dict[str, List[str]],
    idx: Dict[str, Any],
    fk: str,
) -> None:
    cids = chains.get(fk) or []
    if not cids:
        st.warning("No windows for this family.")
        return

    labels: List[str] = []
    cids_plot: List[str] = []
    slug_for_win: List[str] = []
    n_snaps_per_win: List[int] = []
    y_lo: List[float] = []
    y_hi: List[float] = []
    n_lo: List[float] = []
    n_hi: List[float] = []
    hover_y: List[str] = []
    hover_n: List[str] = []

    wn = 0
    for cid in cids:
        snaps = by_cid.get(cid) or []
        if not snaps:
            continue
        ex = window_outcome_book_extremes(snaps)
        has_y = ex["yes_low"] is not None or ex["yes_high"] is not None
        has_n = ex["no_low"] is not None or ex["no_high"] is not None
        if not has_y and not has_n:
            continue
        wn += 1
        info = idx[cid]
        slug = (info.market_slug or cid)[:88]
        labels.append(f"W{wn}")
        cids_plot.append(cid)
        slug_for_win.append(slug)
        n_snaps_per_win.append(len(snaps))

        if has_y:
            yl = float(ex["yes_low"] if ex["yes_low"] is not None else ex["yes_high"] or 0.0)
            yh = float(ex["yes_high"] if ex["yes_high"] is not None else ex["yes_low"] or 0.0)
            if yh < yl:
                yl, yh = yh, yl
            y_lo.append(yl)
            y_hi.append(yh)
            hover_y.append(f"{slug}<br>YES low={yl:.4f} high={yh:.4f}")
        else:
            y_lo.append(0.0)
            y_hi.append(0.0)
            hover_y.append(f"{slug}<br>YES: no book data")

        if has_n:
            nl = float(ex["no_low"] if ex["no_low"] is not None else ex["no_high"] or 0.0)
            nh = float(ex["no_high"] if ex["no_high"] is not None else ex["no_low"] or 0.0)
            if nh < nl:
                nl, nh = nh, nl
            n_lo.append(nl)
            n_hi.append(nh)
            hover_n.append(f"{slug}<br>NO low={nl:.4f} high={nh:.4f}")
        else:
            n_lo.append(0.0)
            n_hi.append(0.0)
            hover_n.append(f"{slug}<br>NO: no book data")

    if not labels:
        st.warning("No snapshot rows for windows in this family.")
        return

    y_h = [max(0.0, y_hi[i] - y_lo[i]) for i in range(len(labels))]
    n_h = [max(0.0, n_hi[i] - n_lo[i]) for i in range(len(labels))]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("YES outcome — min bid → max ask (summary)", "NO outcome — min bid → max ask (summary)"),
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            base=y_lo,
            y=y_h,
            marker_color="#27ae60",
            name="YES range",
            hovertext=hover_y,
            hoverinfo="text",
            width=0.62,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            base=n_lo,
            y=n_h,
            marker_color="#c0392b",
            name="NO range",
            hovertext=hover_n,
            hoverinfo="text",
            width=0.62,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[0, 1], title="Price", row=1, col=1)
    fig.update_yaxes(range=[0, 1], title="Price", row=2, col=1)
    fig.update_xaxes(title="Window (chronological in chain)", tickangle=-45, row=2, col=1)
    fig.update_layout(
        title=f"{fk} — range summary",
        height=520 + min(8, len(labels)) * 2,
        showlegend=False,
        margin=dict(b=120, t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full curves (best bid & ask vs time)**")
    opts = list(range(len(labels)))

    def _fmt_win(i: int) -> str:
        s = slug_for_win[i]
        tail = s if len(s) <= 56 else s[:53] + "…"
        return f"{labels[i]} — {tail} ({n_snaps_per_win[i]} snaps)"

    pick = st.selectbox(
        "Choose window",
        options=opts,
        format_func=_fmt_win,
        key=_streamlit_key("wbcurve_win", fk),
        help="Plots every snapshot in the window: solid = best bid, dashed = best ask.",
    )
    cid_sel = cids_plot[pick]
    snaps_curve = by_cid.get(cid_sel) or []
    _render_window_bid_ask_curves(snaps_curve, fk, labels[pick], slug_for_win[pick])


def _totals_by_buy(
    agg_family: Dict[Tuple[str, float], Dict[str, float]],
    families: List[str],
    buy_levels: List[float],
    margin_cents: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for buy in buy_levels:
        pnl = sum(agg_family.get((fk, buy), {"pnl": 0.0, "dep": 0.0})["pnl"] for fk in families)
        dep = sum(agg_family.get((fk, buy), {"pnl": 0.0, "dep": 0.0})["dep"] for fk in families)
        bc = int(round(buy * 100))
        sc = int(round(min(0.99, buy + margin_cents / 100.0) * 100))
        r = (100.0 * pnl / dep) if dep > 0 else 0.0
        out.append(
            {
                "buy_cents": bc,
                "sell_cents": sc,
                "sum_pnl_usdc": round(pnl, 4),
                "sum_deployed_usdc": round(dep, 4),
                "ret_pct": round(r, 2),
            }
        )
    return out


def main() -> None:
    st.set_page_config(page_title="Hedge backtest (snapshots)", layout="wide")
    st.title("Hedge-cancel snapshot backtest")
    st.caption(
        "Symmetric limit buys on YES & NO, sell = buy + margin. **When either R1 buy fills**, the other buy is "
        "cancelled if still open (YES fill checked before NO on each row). Optional round 2 / multi-cycle. "
        "Settlement uses Gamma when enabled (cached per slug)."
    )

    with st.sidebar:
        st.subheader("Data")
        data_mode = st.radio(
            "Snapshot source",
            options=("monolithic", "split", "sqlite"),
            format_func=lambda m: {
                "monolithic": "Monolithic (crypto_clob_*.jsonl)",
                "split": "Split (one JSONL per family)",
                "sqlite": "SQLite (indexed clob_snapshots.db)",
            }[m],
            horizontal=True,
            key="bt_data_mode",
            help="Split files: split_clob_snapshots_by_family.py. SQLite: migrate_clob_jsonl_to_sqlite.py or --sqlite-db on recorder.",
        )
        sqlite_db_path: Optional[Path] = None
        path_strs: List[str] = []
        selected: List[str] = []
        ms_key = "bt_snapshot_ms_mono"
        if data_mode == "sqlite":
            sqlite_db_path = Path(
                st.text_input(
                    "SQLite database",
                    value=str(DEFAULT_SQLITE_DB),
                    key="bt_sqlite_db",
                    help="Path to clob_snapshots.db (indexed family_key + full JSON payload per row).",
                )
            ).expanduser()
            st.caption("Family list uses fast DISTINCT queries; backtest loads only selected families.")
        elif data_mode == "monolithic":
            clob_dir = Path(
                st.text_input("CLOB snapshots directory", value=str(DEFAULT_CLOB), key="bt_clob_dir")
            ).expanduser()
            paths_sorted = sorted(clob_dir.glob("crypto_clob_*.jsonl")) if clob_dir.is_dir() else []
            path_strs = [str(p) for p in paths_sorted]
            default_pick = path_strs[-2:] if len(path_strs) >= 2 else path_strs
            ms_key = "bt_snapshot_ms_mono"
            selected = st.multiselect(
                "Snapshot files (JSONL)",
                options=path_strs,
                default=[p for p in default_pick if p in path_strs],
                help="Merged in sorted path order.",
                key=ms_key,
            )
        else:
            split_dir = Path(
                st.text_input(
                    "Split families directory",
                    value=str(DEFAULT_SPLIT_FAMILIES),
                    key="bt_split_dir",
                )
            ).expanduser()
            paths_sorted = list_split_family_files(split_dir)
            path_strs = [str(p) for p in paths_sorted]
            default_pick = path_strs
            ms_key = "bt_snapshot_ms_split"
            selected = st.multiselect(
                "Snapshot files (JSONL)",
                options=path_strs,
                default=[p for p in default_pick if p in path_strs],
                help="Merged in sorted path order. One JSONL per family — all selected by default.",
                key=ms_key,
            )

        st.subheader("Markets")
        st.caption("Load families from the selected files, then choose which recurring market types to simulate.")
        if st.button("Load / refresh family list", key="btn_refresh_families"):
            if data_mode == "sqlite":
                dbp = sqlite_db_path
                if dbp is None or not dbp.is_file():
                    st.warning("SQLite file not found — check the path.")
                else:
                    _, count_rows_sqlite, list_family_keys_sqlite, _ = _sqlite_stack()
                    with st.spinner("Reading family keys from database…"):
                        keys = list_family_keys_sqlite(dbp)
                        nrows = count_rows_sqlite(dbp)
                    st.session_state["bt_families_all"] = keys
                    st.session_state["bt_families_files_key"] = ("sqlite", str(dbp.resolve()))
                    st.session_state["bt_families_n_rows"] = nrows
                    st.session_state["bt_picked_families"] = list(keys)
                    st.session_state["bt_rows_all"] = None
                    st.session_state["bt_sqlite_path_resolved"] = str(dbp.resolve())
            elif not selected:
                st.warning("Select snapshot files first.")
            else:
                with st.spinner("Indexing snapshots…"):
                    rows_tmp = _load_rows([Path(p) for p in selected])
                    keys = _family_keys_from_rows(rows_tmp)
                    st.session_state["bt_families_all"] = keys
                    st.session_state["bt_families_files_key"] = (data_mode, tuple(sorted(selected)))
                    st.session_state["bt_families_n_rows"] = len(rows_tmp)
                    st.session_state["bt_picked_families"] = list(keys)
                    st.session_state["bt_rows_all"] = rows_tmp
        fam_all: List[str] = st.session_state.get("bt_families_all") or []
        if data_mode == "sqlite" and sqlite_db_path is not None:
            data_key = ("sqlite", str(sqlite_db_path.resolve()))
        else:
            data_key = (data_mode, tuple(sorted(selected)))
        if fam_all and st.session_state.get("bt_families_files_key") != data_key:
            st.warning("Snapshot source or file selection changed — click **Load / refresh family list** to update.")
        core_names = sorted({_family_core_name(f) for f in fam_all}) if fam_all else []
        name_filter = st.selectbox(
            "Narrow by market name",
            options=(["(all families)"] + core_names) if fam_all else ["(load family list first)"],
            index=0,
            key="bt_fam_name_dropdown",
            help="Choose a name (after the [15M]/[1H] tag) to limit the checklist below, or all families.",
            disabled=not fam_all,
        )
        if name_filter == "(all families)" and fam_all:
            opts = list(fam_all)
        elif name_filter == "(load family list first)":
            opts = []
        elif fam_all:
            opts = [f for f in fam_all if _family_core_name(f) == name_filter]
        else:
            opts = []
        bucket_pick = st.multiselect(
            "Buckets",
            options=["15M", "1H", "Other"],
            default=["15M", "1H", "Other"],
            help="Family keys look like [15M] … or [1H] …. Other = rest.",
        )

        def _bucket_of(fk: str) -> str:
            if fk.startswith("[15M]"):
                return "15M"
            if fk.startswith("[1H]"):
                return "1H"
            return "Other"

        opts_b = [f for f in opts if _bucket_of(f) in bucket_pick]
        if not opts_b and fam_all:
            st.caption("_No families match name + bucket filters — set **Narrow by market name** to (all) or widen **Buckets**._")
        st.multiselect(
            "Market families to run",
            options=opts_b,
            help="Empty selection on Run = error. Pick at least one row.",
            key="bt_picked_families",
            disabled=not fam_all or not opts_b,
        )

        st.subheader("Strategy")
        buy_csv = st.text_input(
            "Buy levels (¢)",
            value="10,20,30,40,50,60,70,80",
            help="Comma-separated cents, or range like 10-80:10",
        )
        margin_cents = int(st.number_input("Margin (¢)", min_value=1, max_value=15, value=3))
        n_cycles = int(st.selectbox("n_cycles", options=(1, 2, 3, 4, 5), index=0))
        cycle_step_cents = int(st.slider("cycle_step_cents", 0, 5, 0))
        size = float(st.number_input("Size (shares per leg)", min_value=0.01, value=5.0, step=1.0))
        fill_policy = st.selectbox("Fill policy", options=("limit", "touch"), index=0)
        apply_settlement = st.checkbox("Apply settlement", value=True)

        st.divider()
        run = st.button("Run backtest", type="primary")

    if data_mode == "sqlite":
        if sqlite_db_path is None or not sqlite_db_path.is_file():
            st.warning("SQLite database not found — set path in sidebar or run migrate_clob_jsonl_to_sqlite.py.")
            return
    elif not selected:
        st.warning("Select at least one snapshot file.")
        return

    fam_all_sidebar: List[str] = st.session_state.get("bt_families_all") or []
    rows_for_charts: Optional[List[Dict[str, Any]]] = st.session_state.get("bt_rows_all")
    if rows_for_charts:
        _render_book_price_ranges_section(rows_for_charts)
    elif data_mode == "sqlite" and fam_all_sidebar:
        st.caption(
            "SQLite: **book charts** need full rows loaded once. Backtest only loads families you select (faster)."
        )
        if st.button("Load snapshot rows for book charts", key="bt_sqlite_load_charts"):
            if sqlite_db_path is not None and sqlite_db_path.is_file():
                load_all_from_sqlite, _, _, _ = _sqlite_stack()
                with st.spinner("Loading all rows from SQLite…"):
                    st.session_state["bt_rows_all"] = load_all_from_sqlite(sqlite_db_path)
                st.rerun()

    if not run and "last_instance_rows" not in st.session_state:
        st.info(
            "Load **family list** (sidebar) to populate book price charts. "
            "Tune parameters and click **Run backtest** for simulation results."
        )
        return

    if run:
        buy_levels = _parse_buy_levels(buy_csv)
        if not buy_levels:
            st.error("No buy levels parsed.")
            return
        if not fam_all_sidebar:
            st.error("Click **Load / refresh family list** in the sidebar first.")
            return
        # `picked_families` comes from widget with key bt_picked_families — read from session
        picked_run = st.session_state.get("bt_picked_families") or []
        if not picked_run:
            st.error("Select at least one market family under **Markets**.")
            return
        if data_mode == "sqlite":
            dbp = Path(
                st.session_state.get("bt_sqlite_path_resolved")
                or st.session_state.get("bt_sqlite_db")
                or str(DEFAULT_SQLITE_DB)
            ).expanduser()
            if not dbp.is_file():
                st.error(f"SQLite file missing: {dbp}")
                return
            _, _, _, load_rows_from_sqlite = _sqlite_stack()
            with st.spinner("Loading snapshots (selected families only)…"):
                rows_all = load_rows_from_sqlite(dbp, family_keys=set(picked_run))
        else:
            paths = [Path(p) for p in selected]
            with st.spinner("Loading snapshots…"):
                rows_all = _load_rows(paths)
        if not rows_all:
            st.error("No rows loaded.")
            return
        st.session_state["bt_rows_all"] = rows_all
        all_families_in_data = set(_family_keys_from_rows(rows_all))
        picked_run = [p for p in picked_run if p in all_families_in_data]
        if not picked_run:
            st.error(
                "No selected families exist in the loaded snapshots. "
                "Click **Load / refresh family list** or adjust your selection."
            )
            return
        fam_filter_set = set(picked_run)
        src_files = (
            [dbp.name]
            if data_mode == "sqlite"
            else [Path(p).name for p in selected]
        )
        st.session_state["rows_meta"] = {
            "data_mode": st.session_state.get("bt_data_mode", "monolithic"),
            "files": src_files,
            "n_rows": len(rows_all),
            "families_run": sorted(fam_filter_set),
            "n_families_run": len(fam_filter_set),
            "buy_levels": buy_levels,
            "margin_cents": margin_cents,
            "n_cycles": n_cycles,
            "cycle_step_cents": cycle_step_cents,
            "size": size,
            "fill_policy": fill_policy,
            "apply_settlement": apply_settlement,
        }
        n_fam = len(fam_filter_set)
        with st.spinner(
            f"Running sim… ({len(rows_all):,} rows × {n_fam} famil(y/ies) × {len(buy_levels)} buy levels)"
        ):
            instance_rows, agg = _run_sweep(
                rows_all,
                buy_levels,
                margin_cents,
                n_cycles,
                cycle_step_cents,
                size,
                fill_policy,
                apply_settlement,
                family_keys_filter=fam_filter_set,
            )
        st.session_state["last_instance_rows"] = instance_rows
        st.session_state["last_agg"] = agg
        st.session_state["last_buy_levels"] = buy_levels
        st.success("Done.")

    meta = st.session_state.get("rows_meta", {})
    instance_rows: List[Dict[str, Any]] = st.session_state.get("last_instance_rows", [])
    agg: Dict[Tuple[str, float], Dict[str, float]] = st.session_state.get("last_agg", {})
    buy_levels: List[float] = st.session_state.get("last_buy_levels", [])

    if not instance_rows or not buy_levels:
        return

    st.subheader("Run summary")
    st.caption("Tables and charts reflect the **last** Run. Change the sidebar and click **Run backtest** again to refresh.")
    st.write(
        {
            "files": meta.get("files"),
            "snapshot_rows": meta.get("n_rows"),
            "families_run": meta.get("families_run"),
            "n_families_run": meta.get("n_families_run"),
            "buy_levels_¢": [int(round(b * 100)) for b in buy_levels],
            "margin_¢": meta.get("margin_cents"),
            "n_cycles": meta.get("n_cycles"),
            "cycle_step_¢": meta.get("cycle_step_cents"),
            "size": meta.get("size"),
            "fill_policy": meta.get("fill_policy"),
            "settlement": meta.get("apply_settlement"),
        }
    )

    families = sorted({r["family"] for r in instance_rows})
    totals = _totals_by_buy(agg, families, buy_levels, meta.get("margin_cents", 3))

    st.subheader("All markets — totals by buy level")
    st.dataframe(totals, use_container_width=True)

    ret_table, pnl_table = _build_family_tables(agg, families, buy_levels)
    st.subheader("By market family — return % (columns = buy ¢)")
    st.dataframe(ret_table, use_container_width=True)
    st.subheader("By market family — sum PnL USDC (columns = buy ¢)")
    st.dataframe(pnl_table, use_container_width=True)

    st.subheader("By market instance (one window = one condition / slug)")
    c1, c2, c3 = st.columns(3)
    with c1:
        fam_pick = st.selectbox("Family filter", options=families)
    with c2:
        buy_pick_c = st.selectbox(
            "Buy level (¢)",
            options=[int(round(b * 100)) for b in buy_levels],
        )
    with c3:
        chart_metric = st.selectbox("Chart metric", options=("pnl_usdc", "ret_pct"), index=0)

    buy_pick = round(buy_pick_c / 100.0, 2)
    inst_filt = [
        r
        for r in instance_rows
        if r["family"] == fam_pick and r["buy_cents"] == buy_pick_c
    ]
    inst_filt.sort(key=lambda x: x["t_first_iso"])
    if not inst_filt:
        st.warning("No rows for this filter.")
        return

    labels = []
    ys = []
    for r in inst_filt:
        slug = r["market_slug"] or r["condition_id"][:12]
        labels.append(slug[-28:] if len(slug) > 28 else slug)
        ys.append(r[chart_metric])

    color = "#2980b9" if chart_metric == "pnl_usdc" else "#8e44ad"
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=ys,
            marker_color=color,
            text=[round(y, 4) for y in ys],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{fam_pick} · buy {buy_pick_c}¢ / sell {buy_pick_c + meta.get('margin_cents', 3)}¢ · {chart_metric}",
        xaxis_title="Market instance (slug suffix)",
        height=max(420, min(900, 180 + 14 * len(labels))),
        xaxis_tickangle=-45,
        margin=dict(b=160),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Instance heatmap (return % per window × buy level)")
    c4, c5 = st.columns(2)
    with c4:
        fam_hm = st.selectbox("Family for heatmap", options=families, key="hm_fam")
    rows_hm = [r for r in instance_rows if r["family"] == fam_hm]
    win_rows: List[Tuple[str, str]] = []
    seen_c: set[str] = set()
    for r in sorted(rows_hm, key=lambda x: x["t_first_iso"]):
        cid = r["condition_id"]
        if cid in seen_c:
            continue
        seen_c.add(cid)
        slug = (r.get("market_slug") or cid) or ""
        ylab = slug[-24:] if len(slug) > 24 else slug
        win_rows.append((cid, ylab))
    buy_cols = [int(round(b * 100)) for b in buy_levels]
    z: List[List[float]] = []
    y_labels: List[str] = []
    for cid, ylab in win_rows:
        y_labels.append(ylab)
        row_z: List[float] = []
        for bc in buy_cols:
            hit = next(
                (x for x in rows_hm if x["condition_id"] == cid and x["buy_cents"] == bc),
                None,
            )
            row_z.append(float(hit["ret_pct"]) if hit else 0.0)
        z.append(row_z)
    fig2 = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[str(c) for c in buy_cols],
            y=y_labels,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Ret %"),
        )
    )
    fig2.update_layout(
        title=f"Return % · {fam_hm}",
        xaxis_title="Buy (¢)",
        yaxis_title="Instance",
        height=max(400, min(1200, 120 + 18 * len(y_labels))),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Instance table (download / filter)")
    show_inst = _filter_instance_rows_for_table(instance_rows, fam_pick, buy_pick_c)
    st.dataframe(show_inst, use_container_width=True, height=360)

    if instance_rows:
        keys = list(instance_rows[0].keys())
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(instance_rows)
        st.download_button(
            "Download full instance grid (CSV)",
            buf.getvalue(),
            file_name="hedge_backtest_instances.csv",
            mime="text/csv",
            key="download_instances_csv",
        )


def _filter_instance_rows_for_table(
    instance_rows: List[Dict[str, Any]],
    fam_pick: str,
    buy_pick_c: int,
) -> List[Dict[str, Any]]:
    """Narrow columns for display."""
    out: List[Dict[str, Any]] = []
    for r in instance_rows:
        if r["family"] != fam_pick or r["buy_cents"] != buy_pick_c:
            continue
        out.append(
            {
                "market_slug": r["market_slug"],
                "t_first": r["t_first_iso"][:19],
                "pnl_usdc": r["pnl_usdc"],
                "deployed_usdc": r["deployed_usdc"],
                "ret_pct": r["ret_pct"],
                "yes_cancelled": r["yes_cancelled"],
                "no_cancelled": r["no_cancelled"],
            }
        )
    return out


if __name__ == "__main__":
    main()
