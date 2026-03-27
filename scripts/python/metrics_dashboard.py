"""
Local trading metrics dashboard. Run from repo root:

  streamlit run scripts/python/metrics_dashboard.py
"""

from __future__ import annotations

import json
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root (parent of `agents/`) — Streamlit does not set PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agents.application.clob_price_history import (
    fetch_binary_window_prices,
    fetch_gamma_market_by_slug,
    simulate_one_shot_dual_limit,
)
from agents.application.gamma_series_history import (
    SeriesWindowRow,
    fetch_event_by_slug,
    fetch_series_events,
    resolved_winner_outcome_label,
    series_id_from_event,
    window_row_from_event,
    window_ts_from_slug,
)

# Bump when series chart cache payload shape changes (invalidates old `gamma_series_cache`).
GAMMA_SERIES_CACHE_VERSION = 3
from agents.application.clob_snapshot_backtest import (
    RecordedMarketInfo,
    index_family_chains,
    index_recorded_markets,
    load_all_snapshot_files,
    merged_yes_ask_series_for_chain,
    rows_for_condition,
)
from agents.application.hedge_cancel_strategy import (
    backtest_hedge_chain_reinvest,
    simulate_symmetric_hedge_cancel_book_snapshots,
    sweep_hedge_cancel_margin_sliding,
)
from agents.application.polymarket_url import slug_from_polymarket_url
from agents.application.run_control import (
    clear_stop_all,
    list_stop_marker_names,
    request_stop_all,
    request_stop_session,
    request_stop_slug,
)

DEFAULT_SESSIONS = REPO_ROOT / "data" / "sessions"
DEFAULT_CLOB_SNAPSHOTS = REPO_ROOT / "data" / "clob_snapshots"
DEFAULT_BACKTEST_LOGS = REPO_ROOT / "data" / "backtest_logs"
LOCAL_KEYS_DIR = REPO_ROOT / "data" / "local"

# Short labels for charts / tables (avoid duplicated prose in the UI).
STRATEGY_SHORT = {
    "hedge_cancel": "Hedge-cancel (symmetric, multi-cycle, sliding spread)",
}
STRATEGY_HELP = {
    "hedge_cancel": (
        "Limit buys on **YES and NO**. Whichever leg completes **buy then sell** first: cancel the other leg’s "
        "buy **only if it never filled**; if it already bought, keep its sell. Then round 2 only on the side "
        "that had its buy cancelled. **Same spread** on both tokens: sell = buy + *m*¢. "
        "Optionally run **1–5 cycles** per market window: after an episode completes, restart on the remaining book. "
        "**Cycle step** (0–5¢): each later cycle adds this to the spread (higher priced-in edge per round-trip vs the prior cycle)."
    ),
}
SAVED_UPLOADED_KEYS = LOCAL_KEYS_DIR / "polymarket_keys.env"


def _sweep_hc_compat(snapshots: List[Dict[str, Any]], **kwargs: Any) -> Any:
    """
    Call sweep_hedge_cancel_margin_sliding with only supported kwargs.
    This avoids runtime mismatch when a long-lived Streamlit process has stale imports.
    """
    sig = inspect.signature(sweep_hedge_cancel_margin_sliding)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return sweep_hedge_cancel_margin_sliding(snapshots, **filtered)


def _backtest_hc_chain_compat(**kwargs: Any) -> Any:
    """Call backtest_hedge_chain_reinvest with supported kwargs only."""
    sig = inspect.signature(backtest_hedge_chain_reinvest)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return backtest_hedge_chain_reinvest(**filtered)


def _init_keys_session_state() -> None:
    if "polymarket_keys_resolved_path" not in st.session_state:
        st.session_state["polymarket_keys_resolved_path"] = ""
    if "poly_keys_path_input" not in st.session_state:
        # Prefer last saved path pointer if present
        pointer = LOCAL_KEYS_DIR / "polymarket_keys_path.txt"
        if pointer.is_file():
            try:
                st.session_state["poly_keys_path_input"] = pointer.read_text(
                    encoding="utf-8"
                ).strip()
            except OSError:
                st.session_state["poly_keys_path_input"] = ""
        else:
            st.session_state["poly_keys_path_input"] = ""


def _render_polymarket_keys_section() -> None:
    _init_keys_session_state()
    LOCAL_KEYS_DIR.mkdir(parents=True, exist_ok=True)

    st.subheader("Polymarket keys (.env)")
    st.caption(
        "**Browse** opens your browser’s file chooser (same as picking a file in a typical file manager). "
        "Use a `.env` with `POLYGON_WALLET_PRIVATE_KEY` and CLOB credentials, or paste a path on this machine."
    )

    uploaded = st.file_uploader(
        "Browse for keys file",
        type=None,
        accept_multiple_files=False,
        help="Click “Browse files” to open the file picker (.env, .txt, or any file with your vars).",
        key="poly_keys_file_uploader",
    )
    if uploaded is not None:
        SAVED_UPLOADED_KEYS.write_bytes(uploaded.getvalue())
        resolved = str(SAVED_UPLOADED_KEYS.resolve())
        st.session_state["polymarket_keys_resolved_path"] = resolved
        st.session_state["poly_keys_path_input"] = resolved
        pointer = LOCAL_KEYS_DIR / "polymarket_keys_path.txt"
        pointer.write_text(resolved + "\n", encoding="utf-8")
        st.success(
            f"Saved **{uploaded.name}** to `{SAVED_UPLOADED_KEYS.relative_to(REPO_ROOT)}`. "
            "Path remembered for this session and stored in `data/local/polymarket_keys_path.txt`."
        )

    manual = st.text_input(
        "Path to keys / .env file on this computer",
        key="poly_keys_path_input",
        placeholder=f"Example: {Path.home() / 'secrets' / 'polymarket.env'}",
    )

    manual_stripped = (manual or "").strip()
    if manual_stripped:
        p = Path(manual_stripped).expanduser()
        if p.is_file():
            st.session_state["polymarket_keys_resolved_path"] = str(p.resolve())
            pointer = LOCAL_KEYS_DIR / "polymarket_keys_path.txt"
            pointer.write_text(str(p.resolve()) + "\n", encoding="utf-8")
            st.markdown(
                "_File found — path saved to `data/local/polymarket_keys_path.txt`._"
            )
        else:
            st.markdown(
                "_That path is not an existing file (check spelling and permissions)._"
            )

    resolved = (
        st.session_state.get("polymarket_keys_resolved_path", "")
        or manual_stripped
    ).strip()
    if resolved:
        rp = Path(resolved).expanduser()
        if rp.is_file():
            st.info(
                f"**Active keys file:** `{rp.resolve()}`  \n"
                f"In a terminal from the repo: `set -a && source '{rp.resolve()}' && set +a`  \n"
                f"Or: `export DOTENV_PATH='{rp.resolve()}'` if your tooling reads that (see project docs)."
            )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def urls_to_cli_list(text: str) -> str:
    parts: List[str] = []
    for line in (text or "").splitlines():
        for chunk in line.split(","):
            c = chunk.strip()
            if c:
                parts.append(c)
    return ",".join(parts)


def estimated_leg_pnl_usdc(rows: List[Dict[str, Any]], side: str) -> float:
    """Sum (sell - buy) * size for completed sells on YES or NO leg."""
    est = 0.0
    for r in rows:
        if r.get("type") != "placed_sell" or r.get("side") != side:
            continue
        bp = r.get("buy_price")
        sp = r.get("price")
        sz = r.get("size")
        if bp is not None and sp is not None and sz is not None:
            est += (float(sp) - float(bp)) * float(sz)
    return round(est, 6)


def index_sessions_by_window_ts(files: List[Path]) -> Dict[int, Dict[str, Any]]:
    """Map window timestamp (slug suffix) -> latest session leg P/L by file mtime."""
    best: Dict[int, tuple[float, Dict[str, Any]]] = {}
    for p in files:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        rows = load_jsonl(p)
        if not rows:
            continue
        sm = next((r for r in rows if r.get("type") == "session_start"), None) or {}
        slug = (sm.get("market_slug") or "").strip()
        ts = window_ts_from_slug(slug)
        if ts is None:
            continue
        payload: Dict[str, Any] = {
            "yes_pnl": estimated_leg_pnl_usdc(rows, "YES"),
            "no_pnl": estimated_leg_pnl_usdc(rows, "NO"),
            "session_file": p.name,
            "market_slug": slug,
            "has_outcome": any(r.get("type") == "outcome" for r in rows),
        }
        prev = best.get(ts)
        if prev is None or mtime >= prev[0]:
            best[ts] = (mtime, payload)
    return {k: v[1] for k, v in best.items()}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    outcome = None
    for r in reversed(rows):
        if r.get("type") == "outcome":
            outcome = r.get("outcome")
            break
    placed_buys = [r for r in rows if r.get("type") == "placed_buy"]
    placed_sells = [r for r in rows if r.get("type") == "placed_sell"]
    wait_fills = [r for r in rows if r.get("type") == "wait_fill"]
    polls = [r for r in rows if r.get("type") == "poll_iteration"]
    est = 0.0
    for r in placed_sells:
        bp = r.get("buy_price")
        sp = r.get("price")
        sz = r.get("size")
        if bp is not None and sp is not None and sz is not None:
            est += (float(sp) - float(bp)) * float(sz)

    ttf: Dict[str, Optional[float]] = {"YES": None, "NO": None}
    t_buy: Dict[str, Optional[float]] = {"YES": None, "NO": None}
    target: Dict[str, float] = {}
    for r in rows:
        if r.get("type") == "placed_buy":
            side = r.get("side")
            if side in t_buy:
                t_buy[side] = float(r["ts_monotonic"])
                target[side] = float(r.get("size", 0))
        if r.get("type") == "wait_fill":
            side = r.get("side")
            if side in ttf and t_buy.get(side) is not None and ttf[side] is None:
                filled = float(r.get("filled", 0))
                tg = target.get(side, 0)
                if tg and filled >= tg:
                    ttf[side] = float(r["ts_monotonic"]) - float(t_buy[side])

    session_meta = next((r for r in rows if r.get("type") == "session_start"), None) or {}
    n_cycles_hint = 1 + len([r for r in rows if r.get("type") == "cycle_start"])

    return {
        "outcome": outcome,
        "n_buys": len(placed_buys),
        "n_sells": len(placed_sells),
        "n_wait_fill_events": len(wait_fills),
        "poll_iterations": len(polls),
        "estimated_pnl_usdc_ex_fees": round(est, 6),
        "seconds_to_full_fill": ttf,
        "market_url": session_meta.get("market_url"),
        "market_slug": session_meta.get("market_slug"),
        "buy_price": session_meta.get("buy_price"),
        "sell_price": session_meta.get("sell_price"),
        "buy_size": session_meta.get("buy_size"),
        "n_cycles_hint": n_cycles_hint,
        "session_id": session_meta.get("session_id"),
    }


def session_looks_active(rows: List[Dict[str, Any]]) -> bool:
    return bool(rows) and not any(r.get("type") == "outcome" for r in rows)


def _series_row_to_dict(w: SeriesWindowRow) -> Dict[str, Any]:
    return {
        "slug": w.slug,
        "end_date_iso": w.end_date_iso,
        "end_dt": w.end_dt.isoformat() if w.end_dt else None,
        "resolution": w.resolution,
        "market_closed": w.market_closed,
        "title": w.title,
    }


def _dict_to_series_row(d: Dict[str, Any]) -> SeriesWindowRow:
    end_dt = None
    if d.get("end_dt"):
        try:
            end_dt = datetime.fromisoformat(str(d["end_dt"]).replace("Z", "+00:00"))
        except ValueError:
            end_dt = None
    return SeriesWindowRow(
        slug=str(d["slug"]),
        end_date_iso=str(d.get("end_date_iso") or ""),
        end_dt=end_dt,
        resolution=d.get("resolution"),
        market_closed=bool(d.get("market_closed")),
        title=str(d.get("title") or ""),
    )


def _render_series_history_section(path_obj: Path, files: List[Path]) -> None:
    st.subheader("Series history & session P/L")

    prev = st.session_state.get("gamma_series_cache")
    if isinstance(prev, dict) and prev.get("v") != GAMMA_SERIES_CACHE_VERSION:
        st.session_state.pop("gamma_series_cache", None)

    st.caption(
        "Load recurring windows for the same product line as a Polymarket **event** URL "
        "(Gamma `series_id`). Data is fetched **newest first**: the count is how many "
        "15m windows to include going **backward from the current cycle**. "
        "Match local JSONL sessions by the numeric suffix in `market_slug` "
        "(e.g. `eth-updown-15m-…` ↔ `eth-up-or-down-15m-…`). "
        "**YES** = first outcome (Up); **NO** = second (Down)."
    )
    example = "https://polymarket.com/event/eth-updown-15m-1774373400"
    series_url = st.text_input(
        "Event URL (or slug)",
        value=example,
        help="Any window from the series works; history loads all windows for that series.",
        key="series_history_event_url",
    )
    lim = st.slider(
        "How many windows (most recent → older)",
        50,
        800,
        300,
        50,
        key="series_history_limit",
        help="Gamma returns the latest cycle first, then fills older ones until this count.",
    )
    c_a, c_b, c_c = st.columns([1, 3, 1])
    with c_a:
        load_series = st.button("Load series from Gamma", type="primary", key="btn_load_series")
    with c_c:
        if st.button("Clear series cache", key="btn_clear_gamma_cache"):
            st.session_state.pop("gamma_series_cache", None)
            st.rerun()
    with c_b:
        show_up = st.checkbox("Show Up (YES) leg P/L", value=True, key="series_show_up_pnl")
        show_down = st.checkbox("Show Down (NO) leg P/L", value=True, key="series_show_down_pnl")

    if load_series:
        raw = (series_url or "").strip()
        if not raw:
            st.warning("Enter a URL or slug.")
        else:
            try:
                slug = slug_from_polymarket_url(raw)
            except ValueError:
                slug = raw.split("/")[-1].split("?")[0]

            event = fetch_event_by_slug(slug)
            if not event:
                st.error(f"No Gamma event found for slug `{slug}`.")
            else:
                sid = series_id_from_event(event)
                if not sid:
                    st.error("Event has no `series` id — not a recurring series page.")
                else:
                    evs = fetch_series_events(
                        sid, limit_total=int(lim), ascending=False
                    )
                    rows_win: List[SeriesWindowRow] = []
                    for e in evs:
                        wr = window_row_from_event(e)
                        if wr:
                            rows_win.append(wr)
                    if rows_win:
                        st.session_state["gamma_series_cache"] = {
                            "v": GAMMA_SERIES_CACHE_VERSION,
                            "rows": [_series_row_to_dict(w) for w in rows_win],
                            "meta": {
                                "series_id": sid,
                                "series_slug": event.get("seriesSlug"),
                                "anchor_slug": event.get("slug"),
                            },
                        }
                    else:
                        st.warning("No windows returned from Gamma.")

    cache = st.session_state.get("gamma_series_cache")
    if not cache:
        st.info("Click **Load series from Gamma** to fetch resolutions and overlay session P/L.")
        return

    meta = cache.get("meta") or {}
    st.success(
        f"Series **{meta.get('series_slug') or meta.get('series_id')}** "
        f"(`series_id={meta.get('series_id')}`). Anchor slug: `{meta.get('anchor_slug')}`."
    )
    st.caption(
        "Chart x-axis: **left** = older windows in this batch, **right** = newest (near current time). "
        "If you still see 2025 dates on the right, click **Clear series cache** and **Load** again."
    )

    rows_win = [_dict_to_series_row(d) for d in cache.get("rows") or []]
    if not rows_win:
        return

    sess_by_ts = index_sessions_by_window_ts(files) if files else {}

    out_x: List[Any] = []
    out_y: List[float] = []
    out_color: List[str] = []
    out_text: List[str] = []
    out_slug: List[str] = []
    up_x: List[Any] = []
    up_y: List[float] = []
    up_custom: List[str] = []
    dn_x: List[Any] = []
    dn_y: List[float] = []
    dn_custom: List[str] = []

    for w in sorted(rows_win, key=lambda r: r.end_date_iso or ""):
        x = w.end_dt if w.end_dt is not None else w.end_date_iso
        out_x.append(x)
        out_y.append(1.0)
        if w.resolution == "Up":
            out_color.append("#2ecc71")
            out_text.append("Up")
        elif w.resolution == "Down":
            out_color.append("#e74c3c")
            out_text.append("Down")
        else:
            out_color.append("#95a5a6")
            out_text.append("open" if not w.market_closed else "?")
        out_slug.append(w.slug)
        ts_key = window_ts_from_slug(w.slug)
        sinfo = sess_by_ts.get(ts_key) if ts_key is not None else None
        if sinfo:
            hover = (
                f"{sinfo['session_file']}<br>"
                f"slug {sinfo['market_slug']}<br>"
                f"Up {sinfo['yes_pnl']} / Down {sinfo['no_pnl']} USDC"
            )
            if show_up:
                up_x.append(x)
                up_y.append(float(sinfo["yes_pnl"]))
                up_custom.append(hover)
            if show_down:
                dn_x.append(x)
                dn_y.append(float(sinfo["no_pnl"]))
                dn_custom.append(hover)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if out_x:
        fig.add_trace(
            go.Scatter(
                x=out_x,
                y=out_y,
                mode="markers",
                name="Outcome",
                marker=dict(size=9, color=out_color, line=dict(width=0)),
                text=out_text,
                hovertemplate="%{x}<br>%{text}<br>%{customdata}<extra></extra>",
                customdata=out_slug,
            ),
            secondary_y=True,
        )
    if show_up and up_x:
        fig.add_trace(
            go.Scatter(
                x=up_x,
                y=up_y,
                mode="markers",
                name="Up (YES) leg P/L",
                marker=dict(size=11, symbol="diamond", color="#1f77b4"),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=up_custom,
            ),
            secondary_y=False,
        )
    if show_down and dn_x:
        fig.add_trace(
            go.Scatter(
                x=dn_x,
                y=dn_y,
                mode="markers",
                name="Down (NO) leg P/L",
                marker=dict(size=11, symbol="square", color="#ff7f0e"),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=dn_custom,
            ),
            secondary_y=False,
        )

    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=40, t=40, b=80),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Session leg P/L (USDC, ex fees)", secondary_y=False)
    fig.update_yaxes(
        title_text="Outcome (colour)",
        range=[0.92, 1.08],
        tickvals=[1.0],
        ticktext=[""],
        showticklabels=False,
        secondary_y=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**Outcome row:** green = Up · red = Down · grey = pending / live — all on the same line."
    )

    table_rows = []
    for w in sorted(rows_win, key=lambda r: r.end_date_iso or "", reverse=True):
        ts_key = window_ts_from_slug(w.slug)
        sinfo = sess_by_ts.get(ts_key) if ts_key is not None else None
        table_rows.append(
            {
                "end": w.end_date_iso,
                "resolution": w.resolution or ("—" if not w.market_closed else "?"),
                "slug": w.slug,
                "yes_pnl": sinfo["yes_pnl"] if sinfo else None,
                "no_pnl": sinfo["no_pnl"] if sinfo else None,
                "session_file": sinfo["session_file"] if sinfo else "",
            }
        )
    st.dataframe(table_rows, use_container_width=True, height=320)

    if path_obj.is_dir() and not files:
        st.caption("_No `*.jsonl` in the sessions directory — P/L markers and table columns will be empty._")


def _settlement_yes_pays_one(
    gamma_market: Optional[Dict[str, Any]],
) -> tuple[bool, Optional[str]]:
    if not gamma_market:
        return True, None
    w = resolved_winner_outcome_label(gamma_market)
    if w == "Up":
        return True, w
    if w == "Down":
        return False, w
    return True, w


def _render_backtest_section() -> None:
    st.subheader("Strategy backtest (one-shot dual limit)")
    st.caption(
        "Mid-price **proxy** from CLOB `prices-history` (not real order-book fills or fees). "
        "**Windows** = how many recent instances of the **same series** as the URL (newest first). "
        "Capital assumed per window: **2 × buy price × shares** (both legs)."
    )
    bt_url = st.text_input(
        "Market URL",
        value="https://polymarket.com/event/eth-updown-15m-1774373400",
        key="bt_market_url",
        help="Any event from the product line; used to resolve series and slugs.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        num_windows = int(
            st.number_input(
                "Windows",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                key="bt_num_windows",
                help="Backtest this many recent market instances (series only).",
            )
        )
    with c2:
        sample_interval_ms = int(
            st.number_input(
                "Sample interval (ms)",
                min_value=500,
                max_value=600_000,
                value=60_000,
                step=500,
                key="bt_sample_ms",
                help=(
                    "Target spacing between CLOB price points; sent as fidelity = ms÷60_000 (minutes). "
                    "For 15m windows the API often caps density (~1 point/min)."
                ),
            )
        )
    with c3:
        pad_sec = int(
            st.number_input(
                "Time padding (seconds)",
                min_value=0,
                max_value=600,
                value=120,
                key="bt_pad",
                help="Extra seconds before/after each window for CLOB history.",
            )
        )

    r1, r2, r3 = st.columns(3)
    with r1:
        buy_price = float(
            st.number_input(
                "Limit buy price",
                min_value=0.01,
                max_value=0.99,
                value=0.45,
                step=0.01,
                format="%.2f",
                key="bt_buy",
            )
        )
    with r2:
        sell_price = float(
            st.number_input(
                "Limit sell price",
                min_value=0.01,
                max_value=0.99,
                value=0.50,
                step=0.01,
                format="%.2f",
                key="bt_sell",
            )
        )
    with r3:
        share_count = float(
            st.number_input(
                "Shares per leg",
                min_value=0.1,
                max_value=float(1_000_000),
                value=5.0,
                step=0.5,
                format="%.2f",
                key="bt_shares",
            )
        )

    run_bt = st.button("Run backtest", type="primary", key="btn_run_backtest")

    if not run_bt:
        return

    raw_u = (bt_url or "").strip()
    if not raw_u:
        st.warning("Enter a market URL.")
        return

    try:
        anchor_slug = slug_from_polymarket_url(raw_u)
    except ValueError:
        st.error("Could not parse slug from URL.")
        return

    event = fetch_event_by_slug(anchor_slug)
    if not event:
        st.error(f"No Gamma event for slug `{anchor_slug}`.")
        return

    sid = series_id_from_event(event)
    slugs: List[str] = []
    if sid and num_windows >= 1:
        evs = fetch_series_events(
            str(sid),
            limit_total=num_windows,
            ascending=False,
        )
        seen: set = set()
        for ev in evs:
            wr = window_row_from_event(ev)
            if wr and wr.slug and wr.slug not in seen:
                seen.add(wr.slug)
                slugs.append(wr.slug)
    if not slugs:
        s = str(event.get("slug") or anchor_slug)
        slugs = [s]

    deployed_per_window = 2.0 * buy_price * share_count
    if deployed_per_window <= 0:
        st.error("Deployed capital must be positive.")
        return

    rows_out: List[Dict[str, Any]] = []
    progress = st.progress(0, text="Fetching price history…")
    total_pnl = 0.0
    windows_run = 0

    for idx, slug in enumerate(slugs):
        progress.progress(
            (idx + 1) / max(len(slugs), 1),
            text=f"Window {idx + 1}/{len(slugs)} `{slug}`",
        )
        row_base: Dict[str, Any] = {"slug": slug, "error": ""}
        try:
            bundle = fetch_binary_window_prices(
                slug,
                sample_interval_ms=sample_interval_ms,
                pad_seconds=pad_sec,
            )
            m = fetch_gamma_market_by_slug(slug)
            yes_1, winner = _settlement_yes_pays_one(m)
            sim = simulate_one_shot_dual_limit(
                bundle.aligned,
                buy_price=buy_price,
                sell_price=sell_price,
                size=share_count,
                yes_expires_at_1=yes_1,
            )
            pnl = float(sim["cash_after_settlement"])
            ret_pct = (100.0 * pnl / deployed_per_window) if deployed_per_window else 0.0
            total_pnl += pnl
            windows_run += 1
            end_label = datetime.fromtimestamp(
                bundle.window_end_unix, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")
            row_base.update(
                {
                    "end (UTC)": end_label,
                    "winner": winner or "—",
                    "P/L (USDC)": round(pnl, 6),
                    "Return (%)": round(ret_pct, 4),
                    "bought Y/N": f"{sim.get('bought_yes')}/{sim.get('bought_no')}",
                    "sold Y/N": f"{sim.get('sold_yes')}/{sim.get('sold_no')}",
                }
            )
        except Exception as exc:
            row_base["error"] = str(exc)
        rows_out.append(row_base)

    progress.empty()

    ok_rows = [r for r in rows_out if not r.get("error")]
    err_n = len(rows_out) - len(ok_rows)
    total_deployed = deployed_per_window * windows_run
    agg_ret_pct = (
        (100.0 * total_pnl / total_deployed) if total_deployed > 0 else 0.0
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Total P/L (USDC)",
        f"{total_pnl:,.4f}",
        help=f"Sum over {windows_run} windows (mid-price sim).",
    )
    m2.metric(
        "Return on capital (%)",
        f"{agg_ret_pct:.2f}%",
        help=f"100 × total P/L ÷ ({windows_run} × {deployed_per_window:,.4f} USDC assumed capital).",
    )
    m3.metric(
        "Windows",
        f"{windows_run} ok / {len(rows_out)}",
        help=f"{err_n} failed" if err_n else "All completed",
    )

    st.dataframe(rows_out, use_container_width=True, height=min(420, 48 + 35 * len(rows_out)))
    if err_n:
        st.warning(f"{err_n} window(s) failed — see **error** column.")


def _render_recorded_clob_backtest_section() -> None:
    st.subheader("Recorded order-book backtest")
    st.caption(
        "Uses **local** JSONL from `record_crypto_clob_snapshots.py` (`crypto_clob_*.jsonl`). "
        "Recurring markets are grouped by **bucket + slug without the trailing epoch** "
        "(e.g. all `…-15m-17…` windows → one **product line**). "
        "**Chart:** YES bid/ask vs time across windows. "
        "**Backtest:** symmetric hedge-cancel per window (sliding NO buy × spread 1–5¢), optional **1–5 cycles** "
        "per window on the remaining book, with **cycle step** (0–5¢) added to the spread each later cycle."
    )
    with st.expander(f"ℹ️ Strategy: {STRATEGY_SHORT['hedge_cancel']}", expanded=False):
        st.markdown(STRATEGY_HELP["hedge_cancel"])
    clob_dir_s = st.text_input(
        "CLOB snapshots directory",
        value=str(DEFAULT_CLOB_SNAPSHOTS),
        key="rec_clob_dir",
        help="Folder containing crypto_clob_YYYY-MM-DD.jsonl files.",
    )
    clob_path = Path(clob_dir_s).expanduser()
    if st.button("Scan / reload snapshot files", type="primary", key="btn_scan_clob_snap"):
        rows = load_all_snapshot_files(clob_path)
        st.session_state["clob_snap_rows"] = rows
        st.session_state["clob_snap_dir_resolved"] = str(clob_path.resolve())
    rows_all: List[Dict[str, Any]] = st.session_state.get("clob_snap_rows") or []
    if not rows_all:
        st.info(
            "Set the directory above (default `data/clob_snapshots`) and click **Scan / reload snapshot files**."
        )
        if clob_path.is_dir():
            nfiles = len(list(clob_path.glob("crypto_clob_*.jsonl")))
            st.caption(f"Found **{nfiles}** `crypto_clob_*.jsonl` file(s) on disk — scan to load.")
        return

    idx = index_recorded_markets(rows_all)
    if not idx:
        st.warning("No valid rows (missing condition_id / token ids).")
        return

    chains = index_family_chains(idx)
    chain_keys = sorted(chains.keys())

    st.success(
        f"Loaded **{len(rows_all)}** snapshot row(s), **{len(idx)}** market window(s) in "
        f"**{len(chains)}** product line(s) from `{st.session_state.get('clob_snap_dir_resolved', clob_path)}`."
    )

    def _chain_label(fk: str) -> str:
        cids = chains[fk]
        return f"{fk} — {len(cids)} window(s)"

    default_sel = chain_keys[: min(5, len(chain_keys))]
    selected_families = st.multiselect(
        "Product lines on one chart (concurrent series)",
        options=chain_keys,
        default=default_sel,
        format_func=_chain_label,
        key="rec_clob_family_multiselect",
        help="Each line merges all time-ordered windows for that product; chart breaks between windows.",
    )
    if not selected_families:
        st.warning("Select at least one product line.")
        return

    show_yes_bid = st.checkbox("Also plot YES bid (per line)", value=False, key="rec_clob_show_bid")

    ha, hb, hc = st.columns(3)
    with ha:
        h_y1 = float(
            st.number_input(
                "YES buy (round 1)",
                0.35,
                0.85,
                0.45,
                0.01,
                key="hc_y1",
            )
        )
        h_y2 = float(
            st.number_input(
                "YES buy (round 2)",
                0.35,
                0.85,
                0.45,
                0.01,
                key="hc_y2",
            )
        )
    with hb:
        h_no_lo = float(
            st.number_input("NO buy min (sweep)", 0.35, 0.85, 0.45, 0.01, key="hc_nlo")
        )
        h_no_hi = float(
            st.number_input("NO buy max (sweep)", 0.35, 0.85, 0.60, 0.01, key="hc_nhi")
        )
    with hc:
        h_n2 = float(
            st.number_input(
                "NO buy (round 2)",
                0.35,
                0.85,
                0.45,
                0.01,
                key="hc_n2",
            )
        )
        h_mc1, h_mc2 = st.slider(
            "Spread (¢), both legs",
            min_value=1,
            max_value=5,
            value=(1, 5),
            key="hc_mc",
        )

    cy1, cy2 = st.columns(2)
    with cy1:
        n_cycles_h = int(
            st.selectbox(
                "Hedge cycles per market window",
                options=(1, 2, 3),
                index=0,
                key="rec_n_cycles",
                help="After a hedge episode completes, restart on the remaining snapshots (same window).",
            )
        )
    with cy2:
        cycle_step_cents_h = int(
            st.slider(
                "Cycle step (¢)",
                min_value=0,
                max_value=5,
                value=0,
                key="rec_cycle_step",
                help="Added to the spread on each subsequent cycle (0 = same spread every cycle).",
            )
        )
    fill_policy_h = "limit"
    st.caption("Fill policy: **limit** (locked)")

    h_size = float(
        st.number_input(
            "Base size (shares per leg)",
            0.1,
            1_000_000.0,
            5.0,
            0.5,
            key="hc_size",
        )
    )
    reinvest = st.checkbox(
        "Reinvest profit into the next window (compound size)",
        value=False,
        key="hc_reinvest",
    )

    settle_mode = st.radio(
        "If Gamma has no winner yet",
        options=("Assume YES pays $1", "Assume NO pays $1", "No settlement (only trade cash)"),
        horizontal=True,
        key="rec_settle_mode",
        help="Applies to any YES/NO shares still held at the end of the recorded series.",
    )

    sell_hint = min(0.99, h_y1 + h_mc1 / 100.0)
    fig = go.Figure()
    palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    vline_ts: List[str] = []
    for fi, fk in enumerate(selected_families):
        color = palette[fi % len(palette)]
        cids = chains.get(fk) or []
        short = fk if len(fk) <= 56 else fk[:53] + "…"
        xs, yb, ya = merged_yes_ask_series_for_chain(rows_all, cids)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ya,
                mode="lines",
                name=f"{short} · YES ask",
                line=dict(color=color),
                connectgaps=False,
            )
        )
        if show_yes_bid:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=yb,
                    mode="lines",
                    name=f"{short} · YES bid",
                    line=dict(color=color, dash="dot"),
                    connectgaps=False,
                )
            )
        for cid in cids:
            snaps = rows_for_condition(rows_all, cid)
            if snaps:
                vline_ts.append(str(snaps[0].get("ts_utc") or ""))

    for yv, name, color in (
        (h_y1, "YES buy (R1)", "#555"),
        (sell_hint, "YES sell (min spread)", "#222"),
    ):
        fig.add_hline(
            y=yv,
            line_dash="dash",
            line_color=color,
            annotation_text=name,
            annotation_position="right",
        )

    uniq_starts = sorted({t for t in vline_ts if t})
    shapes: List[Dict[str, Any]] = []
    for i, t0 in enumerate(uniq_starts):
        if i == 0:
            continue
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=t0,
                y0=0,
                x1=t0,
                y1=1,
                line=dict(color="rgba(128,128,128,0.35)", width=1, dash="dot"),
            )
        )

    fig.update_layout(
        height=460,
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="YES price (book touch)",
        shapes=shapes,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Dotted verticals: start of a **new market window** (first snapshot after a reset). "
        "Gaps in a coloured line separate consecutive windows of the same product."
    )

    run_hedge = st.button(
        "Run hedge backtest on selected lines", type="primary", key="btn_rec_hedge_bt"
    )
    if run_hedge:
        gamma_h: Dict[str, Any] = {}

        def settle_h(info: RecordedMarketInfo) -> tuple[bool, bool]:
            apply_settlement = settle_mode != "No settlement (only trade cash)"
            slug = (info.market_slug or "").strip()
            if slug:
                if slug not in gamma_h:
                    gamma_h[slug] = fetch_gamma_market_by_slug(slug)
                g = gamma_h.get(slug)
            else:
                g = None
            yes_gamma, winner_lbl = _settlement_yes_pays_one(g)
            if winner_lbl:
                return yes_gamma, apply_settlement
            if apply_settlement:
                return (settle_mode == "Assume YES pays $1"), True
            return True, False

        h_agg = _backtest_hc_chain_compat(
            rows_all=rows_all,
            idx=idx,
            chains=chains,
            family_keys=selected_families,
            yes_buy_1=h_y1,
            no_buy_min=h_no_lo,
            no_buy_max=h_no_hi,
            margin_cents_min=h_mc1,
            margin_cents_max=h_mc2,
            yes_buy_2=h_y2,
            no_buy_2=h_n2,
            base_size=h_size,
            reinvest=reinvest,
            settle_each_window=settle_h,
            n_cycles=n_cycles_h,
            cycle_step_cents=cycle_step_cents_h,
            fill_policy=fill_policy_h,
        )
        st.info(
            f"**Hedge backtest:** {len(h_agg.get('per_window') or [])} window(s); "
            f"**{n_cycles_h}** cycle(s)/window · cycle step **{cycle_step_cents_h}**¢ · "
            f"reinvest **{'on' if reinvest else 'off'}** · final size **{h_agg.get('final_size', h_size):.4f}**"
        )
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Total P/L (USDC)", f"{h_agg['total_pnl_usdc']:,.4f}")
        hm2.metric("Return on capital (%)", f"{h_agg['aggregate_return_pct']:.2f}%")
        hm3.metric("Windows", str(len(h_agg.get("per_window") or [])))
        st.dataframe(
            h_agg.get("per_window") or [],
            use_container_width=True,
            height=min(420, 40 + 28 * len(h_agg.get("per_window") or [])),
        )
        st.caption(
            "Book touch; no fees. Settlement per window from Gamma when available, else your radio choice."
        )


def _load_backtest_log_records(log_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not log_dir.is_dir():
        return rows
    for p in sorted(log_dir.glob("*.jsonl")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _hedge_cancel_leaderboard_rows(
    hedge_recs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Latest row per (family, market_slug, n_cycles, cycle_step_cents)."""
    latest: Dict[tuple, Dict[str, Any]] = {}
    for r in hedge_recs:
        fam = str(r.get("family") or "")
        slug = str(r.get("market_slug") or "")
        ts = str(r.get("ts_utc") or "")
        n_cycles = int(r.get("n_cycles") or 1)
        cycle_step_cents = int(r.get("cycle_step_cents") or 0)
        key = (fam, slug, n_cycles, cycle_step_cents)
        prev = latest.get(key)
        if prev is None or ts >= str(prev.get("ts_utc") or ""):
            latest[key] = r
    out: List[Dict[str, Any]] = []
    for (fam, slug, n_cycles, cycle_step_cents), r in latest.items():
        gb = r.get("global_best") or r.get("best") or {}
        slab = str(
            r.get("strategy_label")
            or STRATEGY_SHORT.get(str(r.get("strategy")), "")
            or STRATEGY_SHORT.get("hedge_cancel", "")
        )
        out.append(
            {
                "strategy_label": slab,
                "family": fam,
                "market_slug": slug,
                "n_cycles": n_cycles,
                "cycle_step_cents": cycle_step_cents,
                "margin_pct": float(gb.get("margin_pct") or 0),
                "total_pnl_usdc": float(gb.get("total_pnl_usdc") or 0),
                "no_buy": gb.get("no_buy"),
                "spread_cents": gb.get("margin_cents"),
                "ts_utc": r.get("ts_utc"),
            }
        )
    return out


def _best_strategy_per_market_type(
    rows: List[Dict[str, Any]],
    *,
    sort_key: str,
) -> List[Dict[str, Any]]:
    """
    Pick one best row per market type (family) across all variants.
    """
    best_by_family: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        fam = str(r.get("family") or "")
        if not fam:
            continue
        prev = best_by_family.get(fam)
        if prev is None or float(r.get(sort_key) or 0) > float(prev.get(sort_key) or 0):
            best_by_family[fam] = r
    return sorted(
        best_by_family.values(),
        key=lambda x: float(x.get(sort_key) or 0),
        reverse=True,
    )


def _render_backtest_logs_section() -> None:
    st.subheader("Backtest logs & hedge sweep")
    with st.expander(f"ℹ️ Strategy: {STRATEGY_SHORT['hedge_cancel']}", expanded=False):
        st.markdown(STRATEGY_HELP["hedge_cancel"])

    st.caption(
        "**Continuous logs:** `python scripts/python/run_continuous_backtests.py` → `data/backtest_logs/continuous_*.jsonl`. "
        "Each run sweeps **NO buy** (e.g. 45–60¢) × **spread 1–5¢** on **both** YES and NO (sell = buy + spread). "
        "**Margin %** = 100 × P/L ÷ deployed (sim)."
    )
    log_dir_s = st.text_input(
        "Backtest logs directory",
        value=str(DEFAULT_BACKTEST_LOGS),
        key="bt_log_dir",
    )
    log_path = Path(log_dir_s).expanduser()
    recs = _load_backtest_log_records(log_path)
    hedge_recs = [r for r in recs if r.get("kind") == "hedge_cancel_sweep"]
    if hedge_recs:
        st.markdown("#### Leaderboard — top strategy–market pairs (all cycle variants)")
        lb_rows = _hedge_cancel_leaderboard_rows(hedge_recs)
        lb_sort = st.radio(
            "Sort by",
            options=("margin_pct", "total_pnl_usdc"),
            format_func=lambda x: "Margin % (sim, grid best)" if x == "margin_pct" else "Total P/L (USDC, sim)",
            horizontal=True,
            key="bt_lb_sort",
        )
        lb_sorted = sorted(lb_rows, key=lambda r: r[lb_sort], reverse=True)
        st.dataframe(
            lb_sorted[:200],
            use_container_width=True,
            height=min(480, 40 + 26 * min(len(lb_sorted), 24)),
        )
        st.caption(
            f"One row per **family + market_slug + n_cycles + cycle_step_cents** (latest log **ts** per combo). "
            f"{len(lb_rows)} row(s) from {len(hedge_recs)} sweep line(s). "
            f"Logs without cycle fields count as **1** cycle × **0**¢ step."
        )
        st.markdown("#### Best strategy per market type")
        best_market_rows = _best_strategy_per_market_type(lb_rows, sort_key=lb_sort)
        st.dataframe(
            best_market_rows[:200],
            use_container_width=True,
            height=min(420, 40 + 26 * min(len(best_market_rows), 20)),
        )
        st.caption(
            "One row per market type (**family/product line**); selected by the current "
            "**Sort by** metric across all cycle and step variants."
        )
        st.divider()

    if hedge_recs:
        pairs: Dict[str, List[Dict[str, Any]]] = {}
        for r in hedge_recs:
            strat = str(r.get("strategy") or "unknown")
            fam = str(r.get("family") or "")
            slab = str(r.get("strategy_label") or STRATEGY_SHORT.get("hedge_cancel", strat))
            nc = int(r.get("n_cycles") or 1)
            cs = int(r.get("cycle_step_cents") or 0)
            key = f"{slab} :: {fam} :: {nc} cycles · step {cs}¢"
            pairs.setdefault(key, []).append(r)
        for k in pairs:
            pairs[k].sort(key=lambda x: str(x.get("ts_utc") or ""))
        choice = st.selectbox(
            "Strategy + product (from logs)",
            options=sorted(pairs.keys()),
            key="bt_log_pair",
        )
        prs = pairs[choice]
        latest = prs[-1]
        best = latest.get("global_best") or latest.get("best") or {}
        st.caption(f"**Logged strategy:** `{latest.get('strategy', '')}` · mode `{latest.get('sweep_mode', 'legacy')}`")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Latest margin % (global best in grid)",
            f"{best.get('margin_pct', 0):.2f}%",
        )
        c2.metric(
            "Latest total P/L (USDC, sim)",
            f"{best.get('total_pnl_usdc', 0):,.4f}",
        )
        c3.metric("Grid points (last row)", str(latest.get("grid_points", "—")))
        st.caption(
            f"**ts** {latest.get('ts_utc', '')} · `{latest.get('market_slug', '')}` · "
            f"best NO **{best.get('no_buy', best.get('best_no_buy'))}** / "
            f"**{best.get('no_sell', best.get('best_no_sell'))}** · "
            f"margin **{best.get('margin_cents', '—')}**¢"
        )
        bpm = latest.get("best_per_margin") or []
        if bpm:
            fig_m = go.Figure(
                go.Scatter(
                    x=[r.get("margin_cents") for r in bpm],
                    y=[r.get("margin_pct") for r in bpm],
                    mode="lines+markers",
                    name="Margin %",
                    line=dict(color="#2ecc71", width=2),
                    marker=dict(size=10),
                )
            )
            fig_m.update_layout(
                title="Best margin % vs spread (¢) — last log row",
                xaxis_title="Spread sell − buy (¢), both legs",
                yaxis_title="Margin % (best no_buy in column)",
                height=360,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_m, use_container_width=True)

        hist = [
            {
                "ts_utc": r.get("ts_utc"),
                "margin_pct": (r.get("global_best") or r.get("best") or {}).get("margin_pct"),
                "pnl": (r.get("global_best") or r.get("best") or {}).get("total_pnl_usdc"),
            }
            for r in prs
        ]
        st.dataframe(hist, use_container_width=True, height=min(280, 36 + 28 * len(hist)))
    else:
        st.info("No `hedge_cancel_sweep` rows found yet — run the continuous backtest script.")

    st.divider()
    st.markdown(f"**On-demand hedge sweep** — {STRATEGY_SHORT['hedge_cancel']}")
    clob_d = st.text_input(
        "CLOB snapshots dir (for on-demand)",
        value=str(DEFAULT_CLOB_SNAPSHOTS),
        key="bt_od_clob",
    )
    rows_all_od = load_all_snapshot_files(Path(clob_d).expanduser())
    if not rows_all_od:
        st.caption("Scan snapshots in the **Recorded book backtest** tab first, or point to `data/clob_snapshots`.")
        return
    idx_od = index_recorded_markets(rows_all_od)
    chains_od = index_family_chains(idx_od)
    fk = st.selectbox(
        "Product line",
        options=sorted(chains_od.keys()),
        key="bt_od_family",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        no_lo = float(st.number_input("NO buy min", 0.35, 0.85, 0.45, 0.01, key="bt_od_lo"))
    with c2:
        no_hi = float(st.number_input("NO buy max", 0.35, 0.85, 0.60, 0.01, key="bt_od_hi"))
    with c3:
        mc1, mc2 = st.slider(
            "Spread margin (¢), YES & NO",
            min_value=1,
            max_value=5,
            value=(1, 5),
            key="bt_od_mc",
            help="Each whole cent: sell = buy + spread on both YES and NO",
        )
    y1 = float(st.number_input("YES round-1 buy", 0.35, 0.85, 0.45, 0.01, key="bt_y1"))
    y2y = float(st.number_input("YES round-2 buy (if pivot to YES2)", 0.35, 0.85, 0.45, 0.01, key="bt_y2y"))
    y2n = float(st.number_input("NO round-2 buy (if pivot to NO2)", 0.35, 0.85, 0.45, 0.01, key="bt_y2n"))
    oc1, oc2 = st.columns(2)
    with oc1:
        od_n_cycles = int(
            st.selectbox(
                "Cycles per window",
                options=(1, 2, 3, 4, 5),
                index=0,
                key="bt_od_n_cycles",
            )
        )
    with oc2:
        od_cycle_step = int(
            st.slider(
                "Cycle step (¢)",
                min_value=0,
                max_value=5,
                value=0,
                key="bt_od_cycle_step",
                help="Extra spread on each subsequent cycle within the same window.",
            )
        )
    od_fill_policy = "limit"
    st.caption("On-demand fill policy: **limit** (locked)")

    if st.button("Run hedge sweep", type="primary", key="bt_od_run"):
        cids = chains_od.get(fk) or []
        from collections import defaultdict

        hedge_margins: Dict[int, List[float]] = defaultdict(list)
        hedge_best_margins: List[float] = []
        per_win: List[Dict[str, Any]] = []

        for cid in cids:
            info = idx_od[cid]
            snaps = rows_for_condition(rows_all_od, cid)
            if len(snaps) < 2:
                continue
            m = fetch_gamma_market_by_slug(info.market_slug) if info.market_slug else None
            w = resolved_winner_outcome_label(m) if m else None
            yes_1 = False if w == "Down" else True

            _grid, gbest, bpm = _sweep_hc_compat(
                snaps,
                yes_buy_1=y1,
                yes_buy_2=y2y,
                no_buy_2=y2n,
                no_buy_min=no_lo,
                no_buy_max=no_hi,
                margin_cents_min=mc1,
                margin_cents_max=mc2,
                size=5.0,
                yes_expires_at_1=yes_1,
                apply_settlement=True,
                n_cycles=od_n_cycles,
                cycle_step_cents=od_cycle_step,
                fill_policy=od_fill_policy,
            )
            hm = float(gbest.get("margin_pct") or 0) if gbest else 0.0
            hedge_best_margins.append(hm)
            per_win.append(
                {
                    "window": info.market_slug,
                    "hedge_margin_pct": round(hm, 4),
                    "hedge_no_buy": (gbest or {}).get("no_buy"),
                    "hedge_mc": (gbest or {}).get("margin_cents"),
                    "n_cycles": od_n_cycles,
                    "cycle_step_cents": od_cycle_step,
                }
            )
            if bpm:
                for row in bpm:
                    mc = int(row.get("margin_cents") or 0)
                    hedge_margins[mc].append(float(row.get("margin_pct") or 0))

        if not hedge_best_margins:
            st.warning("No results (need snapshots per window).")
            return

        avg_h_margin = sum(hedge_best_margins) / len(hedge_best_margins)
        fig_cmp = go.Figure(
            data=[
                go.Bar(
                    name=STRATEGY_SHORT["hedge_cancel"] + " (grid best / window)",
                    x=["Avg over windows"],
                    y=[avg_h_margin],
                    marker_color="#e67e22",
                ),
            ]
        )
        fig_cmp.update_layout(
            title="Average margin % — hedge (multi-cycle sweep)",
            yaxis_title="Margin %",
            height=320,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        xs = sorted(hedge_margins.keys())
        ys = [sum(hedge_margins[k]) / len(hedge_margins[k]) for k in xs]
        if xs:
            fig_line = go.Figure(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name="Hedge (avg best per margin)",
                    line=dict(color="#e67e22"),
                )
            )
            fig_line.update_layout(
                title="Hedge: mean margin % vs spread (¢) across windows",
                xaxis_title="Spread (¢)",
                yaxis_title="Mean margin %",
                height=340,
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.dataframe(per_win, use_container_width=True)
        st.caption(
            f"Hedge grid-best mean margin **{avg_h_margin:.2f}%** over **{len(per_win)}** window(s); "
            f"**{od_n_cycles}** cycle(s)/window · cycle step **{od_cycle_step}**¢."
        )


def _render_window_logic_flow_section() -> None:
    st.subheader("Window logic flow")
    st.caption(
        "Per-window hedge decision breakdown with numeric flowchart."
    )

    clob_d = st.text_input(
        "CLOB snapshots dir",
        value=str(DEFAULT_CLOB_SNAPSHOTS),
        key="wf_clob_dir",
    )
    rows_all = load_all_snapshot_files(Path(clob_d).expanduser())
    if not rows_all:
        st.warning("No snapshot rows loaded from this directory.")
        return
    idx = index_recorded_markets(rows_all)
    chains = index_family_chains(idx)
    families = sorted(chains.keys())
    if not families:
        st.warning("No market families found in snapshots.")
        return

    family = st.selectbox("Market type (family)", options=families, key="wf_family")
    cids = chains.get(family) or []
    if not cids:
        st.warning("No windows for selected family.")
        return

    a, b, c = st.columns(3)
    with a:
        yes_buy_1 = float(st.number_input("YES buy R1", 0.35, 0.85, 0.45, 0.01, key="wf_y1"))
        yes_buy_2 = float(st.number_input("YES buy R2", 0.35, 0.85, 0.45, 0.01, key="wf_y2"))
    with b:
        no_buy_2 = float(st.number_input("NO buy R2", 0.35, 0.85, 0.45, 0.01, key="wf_n2"))
        size = float(st.number_input("Shares per leg", 1.0, 1_000_000.0, 5.0, 1.0, key="wf_sz"))
    with c:
        n_cycles = int(st.selectbox("Cycles", options=(1, 2, 3, 4, 5), index=1, key="wf_cycles"))
        cycle_step = int(st.slider("Cycle step (¢)", 0, 5, 3, key="wf_step"))
    fill_policy = "limit"
    st.info("Fill policy is locked to **limit**: executed buy/sell prices use your configured limits.")

    table_rows: List[Dict[str, Any]] = []
    sim_by_slug: Dict[str, Dict[str, Any]] = {}
    for cid in cids:
        info = idx[cid]
        snaps = rows_for_condition(rows_all, cid)
        if not snaps:
            continue
        m = fetch_gamma_market_by_slug(info.market_slug) if info.market_slug else None
        w = resolved_winner_outcome_label(m) if m else None
        yes_1 = False if w == "Down" else True
        _, gb, _ = _sweep_hc_compat(
            snaps,
            yes_buy_1=yes_buy_1,
            yes_buy_2=yes_buy_2,
            no_buy_2=no_buy_2,
            no_buy_min=0.45,
            no_buy_max=0.60,
            margin_cents_min=1,
            margin_cents_max=5,
            size=size,
            yes_expires_at_1=yes_1,
            apply_settlement=True,
            n_cycles=n_cycles,
            cycle_step_cents=cycle_step,
            fill_policy=fill_policy,
        )
        if not gb:
            continue
        sim = simulate_symmetric_hedge_cancel_book_snapshots(
            snaps,
            yes_buy_1=yes_buy_1,
            no_buy=float(gb.get("no_buy") or 0.45),
            spread=float(gb.get("spread") or (float(gb.get("margin_cents") or 0) / 100.0)),
            yes_buy_2=yes_buy_2,
            no_buy_2=no_buy_2,
            size=size,
            yes_expires_at_1=yes_1,
            apply_settlement=True,
            n_cycles=n_cycles,
            cycle_step_cents=cycle_step,
            fill_policy=fill_policy,
        )
        slug = str(info.market_slug or cid)
        sim_by_slug[slug] = sim
        ybq = float(sim.get("yes_buy_qty") or 0)
        ybn = float(sim.get("yes_buy_notional") or 0)
        ysq = float(sim.get("yes_sell_qty") or 0)
        ysn = float(sim.get("yes_sell_notional") or 0)
        nbq = float(sim.get("no_buy_qty") or 0)
        nbn = float(sim.get("no_buy_notional") or 0)
        nsq = float(sim.get("no_sell_qty") or 0)
        nsn = float(sim.get("no_sell_notional") or 0)
        # Defensive fallback: if top-level aggregation is missing, derive from per-cycle rows.
        if (ybq + ysq + nbq + nsq) == 0.0 and (sim.get("per_cycle") or []):
            pcs = sim.get("per_cycle") or []
            ybq = sum(float(c.get("yes_buy_qty") or 0) for c in pcs)
            ybn = sum(float(c.get("yes_buy_notional") or 0) for c in pcs)
            ysq = sum(float(c.get("yes_sell_qty") or 0) for c in pcs)
            ysn = sum(float(c.get("yes_sell_notional") or 0) for c in pcs)
            nbq = sum(float(c.get("no_buy_qty") or 0) for c in pcs)
            nbn = sum(float(c.get("no_buy_notional") or 0) for c in pcs)
            nsq = sum(float(c.get("no_sell_qty") or 0) for c in pcs)
            nsn = sum(float(c.get("no_sell_notional") or 0) for c in pcs)
        table_rows.append(
            {
                "market_slug": slug,
                "yes_buy_qty": ybq,
                "yes_buy_px": round(ybn / ybq, 6) if ybq > 0 else None,
                "yes_sell_qty": ysq,
                "yes_sell_px": round(ysn / ysq, 6) if ysq > 0 else None,
                "no_buy_qty": nbq,
                "no_buy_px": round(nbn / nbq, 6) if nbq > 0 else None,
                "no_sell_qty": nsq,
                "no_sell_px": round(nsn / nsq, 6) if nsq > 0 else None,
                "cancelled": bool(sim.get("yes_buy_cancelled") or sim.get("no_buy_cancelled")),
                "spread_cents": gb.get("margin_cents"),
                "no_buy_limit": gb.get("no_buy"),
                "cycles_run": sim.get("n_cycles_run", 1),
                "end_profit_dollars": round(float(sim.get("cash_after_settlement") or 0), 6),
            }
        )

    if not table_rows:
        st.warning("No windows produced results for this configuration.")
        return

    st.dataframe(table_rows, use_container_width=True, height=360)
    selected_slug = st.selectbox(
        "Window to inspect",
        options=[r["market_slug"] for r in table_rows],
        key="wf_window_slug",
    )
    row = next(r for r in table_rows if r["market_slug"] == selected_slug)
    sim = sim_by_slug[selected_slug]
    cancel_text = "Yes" if row["cancelled"] else "No"
    y_buy_text = f"YES buy: {row['yes_buy_qty']} @ {row['yes_buy_px']}"
    n_buy_text = f"NO buy: {row['no_buy_qty']} @ {row['no_buy_px']}"
    y_sell_text = f"YES sell: {row['yes_sell_qty']} @ {row['yes_sell_px']}"
    n_sell_text = f"NO sell: {row['no_sell_qty']} @ {row['no_sell_px']}"
    flow = f"""
digraph G {{
  rankdir=TB;
  node [shape=box, style=rounded];
  A [label="Start window\\n{selected_slug}"];
  B [label="Place limits\\nYES buy={yes_buy_1:.2f}\\nNO buy={row['no_buy_limit']:.2f}\\nspread={int(row['spread_cents'])}c"];
  C [label="{y_buy_text}\\n{n_buy_text}"];
  D [label="{y_sell_text}\\n{n_sell_text}"];
  E [label="Any buy cancelled? {cancel_text}\\nphase={sim.get('phase')}"];
  F [label="End profit\\n${row['end_profit_dollars']:.6f}"];
  A -> B -> C -> D -> E -> F;
}}
"""
    st.graphviz_chart(flow)


def main() -> None:
    st.set_page_config(page_title="One-shot strategy metrics", layout="wide")
    st.title("Trading session metrics")
    st.caption("Estimated P/L excludes fees. Read from JSONL session logs.")

    sessions_dir = st.sidebar.text_input(
        "Sessions directory",
        value=str(DEFAULT_SESSIONS),
        help="Folder containing *.jsonl files",
    )
    path_obj = Path(sessions_dir).expanduser()
    if not path_obj.is_dir():
        st.warning(
            "Sessions directory does not exist yet; create it when you run the bot, "
            "or point the sidebar to an existing folder."
        )
        files = []
    else:
        files = sorted(path_obj.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    choice = st.sidebar.selectbox(
        "Session file (detail)",
        options=[str(p) for p in files] if files else ["(none)"],
    )
    auto = st.sidebar.checkbox("Auto-refresh every 5s", value=False)

    if auto:
        import time

        time.sleep(5)
        st.rerun()

    tab_dash, tab_clob, tab_btlog, tab_wflow = st.tabs(
        ["Dashboard", "Recorded book backtest", "Backtest logs & comparison", "Window logic flow"]
    )

    with tab_clob:
        _render_recorded_clob_backtest_section()

    with tab_btlog:
        _render_backtest_logs_section()

    with tab_dash:
        _render_dashboard_main(path_obj, files, choice)

    with tab_wflow:
        _render_window_logic_flow_section()


def _render_dashboard_main(path_obj: Path, files: List[Path], choice: str) -> None:
    st.subheader("Market links")
    pasted_urls = st.text_area(
        "Paste Polymarket market URLs here (one per line or comma-separated). "
        "Use the snippet below with your terminal to start the bot.",
        height=180,
        placeholder="https://polymarket.com/event/your-market-slug\nhttps://polymarket.com/event/another-market",
        key="pasted_market_urls",
        label_visibility="visible",
    )
    cli_urls = urls_to_cli_list(pasted_urls)
    st.code(
        f'PYTHONPATH=. python scripts/python/cli.py run-one-shot-dual-limit \\\n  --market-urls "{cli_urls}" \\\n  --buy-price 0.45 --sell-price 0.50 --buy-size 5',
        language="bash",
    )

    _render_series_history_section(path_obj, files)

    _render_backtest_section()

    _render_polymarket_keys_section()

    st.subheader("Stop running strategies")
    st.caption(
        "Creates files under **data/control/** that the bot checks every poll. "
        "Run the trading process from this repository root so paths match."
    )
    c_stop1, c_stop2, c_stop3 = st.columns(3)
    with c_stop1:
        if st.button("Stop ALL active markets", type="primary", key="btn_stop_all"):
            request_stop_all()
            st.success("Posted **STOP_ALL**. Each running loop exits on its next poll.")
    with c_stop2:
        if st.button("Clear STOP_ALL", key="btn_clear_stop_all"):
            clear_stop_all()
            st.info("Removed **STOP_ALL** flag.")
    with c_stop3:
        markers = list_stop_marker_names()
        st.caption(f"Control files: {', '.join(markers) if markers else 'none'}")

    if files:
        st.subheader("Sessions that look still running (no outcome row yet)")
        active_rows = []
        for p in files[:30]:
            rws = load_jsonl(p)
            if not session_looks_active(rws):
                continue
            sm = next((r for r in rws if r.get("type") == "session_start"), {}) or {}
            active_rows.append(
                {
                    "file": p.name,
                    "session_id": sm.get("session_id") or p.stem,
                    "slug": sm.get("market_slug") or "",
                    "url": (sm.get("market_url") or "")[:60],
                }
            )
        if active_rows:
            for row in active_rows:
                sid = str(row["session_id"])
                slug = str(row["slug"])
                ca, cb, cc = st.columns([3, 1, 1])
                with ca:
                    st.markdown(
                        f"**{row['file']}** · `{sid}` · "
                        f"{row['url'] or '—'} · slug `{slug or '—'}`"
                    )
                with cb:
                    if st.button("Stop session", key=f"stop_sess_{row['file']}"):
                        request_stop_session(sid)
                        st.success("Stop file written for this session.")
                with cc:
                    if slug and st.button("Stop slug", key=f"stop_slug_{row['file']}"):
                        request_stop_slug(slug)
                        st.success("Stop file written for this slug.")
        else:
            st.write("_No open sessions detected (each JSONL already has an outcome, or no logs yet)._")

    if not files or choice == "(none)":
        st.info(
            "No session JSONL files yet (or none selected). Paste URLs above, run the CLI, "
            "then pick a file in the sidebar for the detail view."
        )
        return

    rows = load_jsonl(Path(choice))
    if not rows:
        st.error("Empty or invalid file.")
        return

    meta = summarize(rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outcome", meta["outcome"] or "—")
    c2.metric("Est. P/L (USDC, ex fees)", meta["estimated_pnl_usdc_ex_fees"])
    c3.metric("Poll iterations", meta["poll_iterations"])
    c4.metric("BUY / SELL orders", f"{meta['n_buys']} / {meta['n_sells']}")

    sid_detail = meta.get("session_id") or Path(choice).stem
    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button("Stop this session (detail view)", key="stop_detail_session"):
            request_stop_session(str(sid_detail))
            st.success("Stop file written for this session.")
    with d2:
        if meta.get("market_slug") and st.button(
            "Stop this market slug",
            key="stop_detail_slug",
        ):
            request_stop_slug(str(meta["market_slug"]))
            st.success("Stop file written for this slug.")
    with d3:
        if st.button("Stop ALL markets", key="stop_detail_all"):
            request_stop_all()
            st.success("STOP_ALL posted.")

    if meta.get("market_url") or meta.get("market_slug"):
        st.write(
            f"**Market:** {meta.get('market_url') or '—'}  |  slug: `{meta.get('market_slug') or '—'}`  |  "
            f"limits: buy **{meta.get('buy_price')}** / sell **{meta.get('sell_price')}**  size **{meta.get('buy_size')}**  |  "
            f"cycles (incl. session): **{meta.get('n_cycles_hint')}**  id: `{sid_detail}`"
        )

    st.subheader("Wait to full fill (monotonic seconds after BUY)")
    wf_cols = st.columns(2)
    wf_cols[0].write(f"**YES:** {meta['seconds_to_full_fill']['YES']}")
    wf_cols[1].write(f"**NO:** {meta['seconds_to_full_fill']['NO']}")

    st.subheader("Orders")
    order_rows = [r for r in rows if r.get("type") in ("placed_buy", "placed_sell")]
    if order_rows:
        st.dataframe(
            [
                {
                    "ts_utc": r.get("ts_utc"),
                    "kind": r.get("type"),
                    "side": r.get("side"),
                    "price": r.get("price"),
                    "size": r.get("size"),
                    "order_id": r.get("order_id"),
                }
                for r in order_rows
            ],
            use_container_width=True,
        )

    st.subheader("Fill progression")
    fill_rows = [r for r in rows if r.get("type") == "wait_fill"]
    if fill_rows:
        chart_data = [
            {
                "i": i,
                "side": r.get("side"),
                "filled": float(r.get("filled", 0)),
                "target": float(r.get("target", 0)),
            }
            for i, r in enumerate(fill_rows)
        ]
        st.dataframe(chart_data, use_container_width=True)


if __name__ == "__main__":
    main()
