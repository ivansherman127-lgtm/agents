import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, cast

import typer
from devtools import pprint

from agents.polymarket.polymarket import Polymarket
from agents.connectors.chroma import PolymarketRAG
from agents.connectors.news import News
from agents.application.trade import Trader
from agents.application.executor import Executor
from agents.application.creator import Creator
from agents.application.dry_run_polymarket import DryRunPolymarket, FillMode
from agents.application.session_metrics import TradeSessionMetrics
from agents.application.clob_price_history import (
    fetch_binary_window_prices,
    fetch_gamma_market_by_slug,
    simulate_one_shot_dual_limit,
)
from agents.application.gamma_series_history import resolved_winner_outcome_label
from agents.application.polymarket_url import (
    resolve_polymarket_market_url,
    slug_from_polymarket_url,
)

app = typer.Typer()
polymarket = Polymarket()
newsapi_client = News()
polymarket_rag = PolymarketRAG()


@app.command()
def get_all_markets(limit: int = 5, sort_by: str = "spread") -> None:
    """
    Query Polymarket's markets
    """
    print(f"limit: int = {limit}, sort_by: str = {sort_by}")
    markets = polymarket.get_all_markets()
    markets = polymarket.filter_markets_for_trading(markets)
    if sort_by == "spread":
        markets = sorted(markets, key=lambda x: x.spread, reverse=True)
    markets = markets[:limit]
    pprint(markets)


@app.command()
def get_relevant_news(keywords: str) -> None:
    """
    Use NewsAPI to query the internet
    """
    articles = newsapi_client.get_articles_for_cli_keywords(keywords)
    pprint(articles)


@app.command()
def get_all_events(limit: int = 5, sort_by: str = "number_of_markets") -> None:
    """
    Query Polymarket's events
    """
    print(f"limit: int = {limit}, sort_by: str = {sort_by}")
    events = polymarket.get_all_events()
    events = polymarket.filter_events_for_trading(events)
    if sort_by == "number_of_markets":
        events = sorted(events, key=lambda x: len(x.markets), reverse=True)
    events = events[:limit]
    pprint(events)


@app.command()
def create_local_markets_rag(local_directory: str) -> None:
    """
    Create a local markets database for RAG
    """
    polymarket_rag.create_local_markets_rag(local_directory=local_directory)


@app.command()
def query_local_markets_rag(vector_db_directory: str, query: str) -> None:
    """
    RAG over a local database of Polymarket's events
    """
    response = polymarket_rag.query_local_markets_rag(
        local_directory=vector_db_directory, query=query
    )
    pprint(response)


@app.command()
def ask_superforecaster(event_title: str, market_question: str, outcome: str) -> None:
    """
    Ask a superforecaster about a trade
    """
    print(
        f"event: str = {event_title}, question: str = {market_question}, outcome (usually yes or no): str = {outcome}"
    )
    executor = Executor()
    response = executor.get_superforecast(
        event_title=event_title, market_question=market_question, outcome=outcome
    )
    print(f"Response:{response}")


@app.command()
def create_market() -> None:
    """
    Format a request to create a market on Polymarket
    """
    c = Creator()
    market_description = c.one_best_market()
    print(f"market_description: str = {market_description}")


@app.command()
def ask_llm(user_input: str) -> None:
    """
    Ask a question to the LLM and get a response.
    """
    executor = Executor()
    response = executor.get_llm_response(user_input)
    print(f"LLM Response: {response}")


@app.command()
def ask_polymarket_llm(user_input: str) -> None:
    """
    What types of markets do you want trade?
    """
    executor = Executor()
    response = executor.get_polymarket_llm(user_input=user_input)
    print(f"LLM + current markets&events response: {response}")


@app.command()
def run_autonomous_trader() -> None:
    """
    Let an autonomous system trade for you.
    """
    trader = Trader()
    trader.one_best_trade()


def _parse_market_urls(urls_option: str) -> List[str]:
    return [u.strip() for u in urls_option.split(",") if u.strip()]


def _slug_tail(s: str, max_len: int = 24) -> str:
    tail = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return tail[:max_len] if tail else "market"


def _job_worker(
    yes_token_id: str,
    no_token_id: str,
    market_slug: str,
    market_url: str,
    buy_price: float,
    sell_price: float,
    buy_size: float,
    cycles: int,
    poll_interval_seconds: int,
    max_runtime_seconds: int,
    dry_run: bool,
    dry_fill_mode: str,
    dry_delay_polls: int,
    session_id: str,
    sessions_dir: str,
) -> None:
    if dry_fill_mode not in ("instant", "delayed", "never"):
        raise typer.BadParameter("dry_fill_mode must be one of: instant, delayed, never")

    metrics = TradeSessionMetrics(
        session_id=session_id,
        sessions_dir=sessions_dir,
    )

    if dry_run:
        trader = Trader(
            polymarket=DryRunPolymarket(
                fill_mode=cast(FillMode, dry_fill_mode),
                delay_polls=dry_delay_polls,
                market_open=True,
            )
        )
    else:
        trader = Trader()

    trader.run_fixed_market_multi_cycle(
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        market_slug=market_slug,
        buy_price=buy_price,
        buy_size=buy_size,
        sell_price=sell_price,
        poll_interval_seconds=poll_interval_seconds,
        max_runtime_seconds=max_runtime_seconds,
        metrics=metrics,
        cycles=cycles,
        market_url=market_url,
    )
    print(
        f"[METRICS] market_slug={market_slug!r} session_id={metrics.session_id} dir={sessions_dir}"
    )


def _job_worker_hedge_cancel(
    yes_token_id: str,
    no_token_id: str,
    market_slug: str,
    market_url: str,
    yes_buy_1: float,
    yes_sell_1: float,
    no_buy_price: float,
    yes_buy_2: float,
    yes_sell_2: float,
    buy_size: float,
    poll_interval_seconds: int,
    max_runtime_seconds: int,
    dry_run: bool,
    dry_fill_mode: str,
    dry_delay_polls: int,
    session_id: str,
    sessions_dir: str,
) -> None:
    if dry_fill_mode not in ("instant", "delayed", "never"):
        raise typer.BadParameter("dry_fill_mode must be one of: instant, delayed, never")

    metrics = TradeSessionMetrics(
        session_id=session_id,
        sessions_dir=sessions_dir,
    )

    if dry_run:
        trader = Trader(
            polymarket=DryRunPolymarket(
                fill_mode=cast(FillMode, dry_fill_mode),
                delay_polls=dry_delay_polls,
                market_open=True,
            )
        )
    else:
        trader = Trader()

    trader.run_fixed_market_hedge_cancel(
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        market_slug=market_slug,
        yes_buy_1=yes_buy_1,
        yes_sell_1=yes_sell_1,
        no_buy_price=no_buy_price,
        yes_buy_2=yes_buy_2,
        yes_sell_2=yes_sell_2,
        buy_size=buy_size,
        poll_interval_seconds=poll_interval_seconds,
        max_runtime_seconds=max_runtime_seconds,
        metrics=metrics,
        market_url=market_url,
    )
    print(
        f"[METRICS] hedge_cancel market_slug={market_slug!r} session_id={metrics.session_id} dir={sessions_dir}"
    )


@app.command()
def run_hedge_cancel_dual_limit(
    market_urls: str = typer.Option(
        "",
        "--market-urls",
        help="Comma-separated Polymarket market URLs (one market).",
    ),
    yes_buy_1: float = typer.Option(0.45, "--yes-buy-1"),
    yes_sell_1: float = typer.Option(0.50, "--yes-sell-1"),
    no_buy_price: float = typer.Option(0.45, "--no-buy-price", help="NO leg buy; sell = this + 0.05"),
    yes_buy_2: float = typer.Option(0.45, "--yes-buy-2", help="Second YES round after NO buy cancel"),
    yes_sell_2: float = typer.Option(0.50, "--yes-sell-2"),
    buy_size: float = typer.Option(5.0, "--buy-size"),
    poll_interval_seconds: int = 3,
    max_runtime_seconds: int = 3600,
    dry_run: bool = False,
    dry_fill_mode: str = "delayed",
    dry_delay_polls: int = 2,
    session_id: str = "",
    sessions_dir: str = "data/sessions",
) -> None:
    """
    Hedge-cancel strategy: when YES round-1 sell fills, cancel NO buy if NO had zero fill;
    else complete NO. After cancel, YES round-2 at --yes-buy-2 / --yes-sell-2.
    """
    url_list = _parse_market_urls(market_urls)
    if len(url_list) != 1:
        raise typer.BadParameter("Provide exactly one --market-urls for hedge-cancel.")
    u0 = url_list[0]
    yt, nt, canon = resolve_polymarket_market_url(u0)
    sid = session_id or str(uuid.uuid4())
    _job_worker_hedge_cancel(
        yes_token_id=yt,
        no_token_id=nt,
        market_slug=canon,
        market_url=u0,
        yes_buy_1=yes_buy_1,
        yes_sell_1=yes_sell_1,
        no_buy_price=no_buy_price,
        yes_buy_2=yes_buy_2,
        yes_sell_2=yes_sell_2,
        buy_size=buy_size,
        poll_interval_seconds=poll_interval_seconds,
        max_runtime_seconds=max_runtime_seconds,
        dry_run=dry_run,
        dry_fill_mode=dry_fill_mode,
        dry_delay_polls=dry_delay_polls,
        session_id=sid,
        sessions_dir=sessions_dir,
    )


@app.command()
def run_one_shot_dual_limit(
    yes_token_id: str = typer.Argument(
        "",
        help="YES outcome clob token id (omit if using --market-urls).",
    ),
    no_token_id: str = typer.Argument(
        "",
        help="NO outcome clob token id (omit if using --market-urls).",
    ),
    market_urls: str = typer.Option(
        "",
        "--market-urls",
        help='Comma-separated Polymarket market page URLs (e.g. "https://polymarket.com/event/...").',
    ),
    market_slug: str = typer.Option(
        "",
        "--market-slug",
        help="Gamma slug for market-open checks (optional if using --market-urls; inferred from URLs).",
    ),
    buy_price: float = typer.Option(0.45, "--buy-price", help="Limit buy price per share."),
    sell_price: float = typer.Option(0.50, "--sell-price", help="Limit sell price per share."),
    buy_size: float = typer.Option(5.0, "--buy-size", help="Shares per leg per cycle."),
    cycles: int = typer.Option(
        1,
        "--cycles",
        help="How many full buy/wait/sell rounds to run per market (sequential).",
        min=1,
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="When multiple markets (--market-urls), run them concurrently (thread per market).",
    ),
    poll_interval_seconds: int = 3,
    max_runtime_seconds: int = 3600,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate orders and fills without calling the live CLOB API.",
    ),
    dry_fill_mode: str = typer.Option(
        "instant",
        "--dry-fill-mode",
        help="Dry-run only: instant | delayed | never.",
    ),
    dry_delay_polls: int = typer.Option(
        2,
        "--dry-delay-polls",
        help="Dry-run delayed mode: report full fill after this many checks per BUY order.",
    ),
    session_id: str = typer.Option(
        "",
        "--session-id",
        help="Base metrics session id (one UUID per market is appended for multi-market runs).",
    ),
    sessions_dir: str = typer.Option(
        "data/sessions",
        "--sessions-dir",
        help="Directory where session metrics JSONL files are written.",
    ),
) -> None:
    """
    Dual-outcome limit strategy: BUY both legs, then SELL after full fill (per cycle).

    Provide either --market-urls (one or more comma-separated Polymarket links) or two token id arguments.
    """
    if dry_fill_mode not in ("instant", "delayed", "never"):
        raise typer.BadParameter("dry_fill_mode must be one of: instant, delayed, never")

    url_list = _parse_market_urls(market_urls)
    jobs: List[Tuple[str, str, str, str]] = []

    if url_list:
        for url in url_list:
            yt, nt, canon = resolve_polymarket_market_url(url)
            slug_for_api = canon
            jobs.append((yt, nt, slug_for_api, url))
    else:
        if not yes_token_id or not no_token_id:
            raise typer.BadParameter(
                "Pass YES and NO token ids as arguments, or set --market-urls."
            )
        slug_use = market_slug.strip() or ""
        jobs.append((yes_token_id, no_token_id, slug_use, ""))

    base_session = session_id or str(uuid.uuid4())

    def launch(
        job_tuple: Tuple[str, str, str, str], idx: int, total: int
    ) -> None:
        yt, nt, slug, url = job_tuple
        sid = (
            base_session
            if total == 1
            else f"{base_session}-{_slug_tail(slug or yt)}-{idx}"
        )
        _job_worker(
            yes_token_id=yt,
            no_token_id=nt,
            market_slug=slug,
            market_url=url,
            buy_price=buy_price,
            sell_price=sell_price,
            buy_size=buy_size,
            cycles=cycles,
            poll_interval_seconds=poll_interval_seconds,
            max_runtime_seconds=max_runtime_seconds,
            dry_run=dry_run,
            dry_fill_mode=dry_fill_mode,
            dry_delay_polls=dry_delay_polls,
            session_id=sid,
            sessions_dir=sessions_dir,
        )

    total_j = len(jobs)
    if parallel and total_j > 1:
        with ThreadPoolExecutor(max_workers=total_j) as pool:
            futures = [
                pool.submit(launch, job, i, total_j)
                for i, job in enumerate(jobs, start=1)
            ]
            for fut in as_completed(futures):
                fut.result()
    else:
        for i, job in enumerate(jobs, start=1):
            launch(job, i, total_j)


def _slug_from_url_or_slug(market_url: str, slug: str) -> str:
    u, s = market_url.strip(), slug.strip()
    if u:
        return slug_from_polymarket_url(u)
    if s:
        return s
    raise typer.BadParameter("Pass --market-url or --slug.")


@app.command()
def fetch_window_prices(
    market_url: str = typer.Option("", "--market-url", help="Polymarket event/market URL."),
    slug: str = typer.Option("", "--slug", help="Gamma market slug (if no URL)."),
    out: str = typer.Option(
        "",
        "--out",
        help="Write JSON bundle here (default: stdout).",
    ),
    sample_ms: int = typer.Option(
        60_000,
        "--sample-ms",
        min=1,
        help="Target spacing between price samples (milliseconds); CLOB uses fidelity = ms/60_000 minutes.",
    ),
    pad_seconds: int = typer.Option(
        120,
        "--pad-seconds",
        min=0,
        help="Extend [start,end] on both sides when calling prices-history.",
    ),
) -> None:
    """
    Download YES/NO share price samples (CLOB ``prices-history``) for one market window.

    JSON includes ``yes_history``, ``no_history`` (each [{t,p}]), and ``aligned``
    rows with forward-filled ``yes_p`` / ``no_p`` for backtests.
    """
    slug_use = _slug_from_url_or_slug(market_url, slug)
    bundle = fetch_binary_window_prices(
        slug_use, sample_interval_ms=sample_ms, pad_seconds=pad_seconds
    )
    payload = bundle.to_json_obj()
    text = json.dumps(payload, indent=2)
    if out.strip():
        Path(out).expanduser().write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)


@app.command()
def backtest_one_shot_from_prices(
    market_url: str = typer.Option("", "--market-url"),
    slug: str = typer.Option("", "--slug"),
    buy_price: float = typer.Option(0.45, "--buy-price"),
    sell_price: float = typer.Option(0.50, "--sell-price"),
    buy_size: float = typer.Option(5.0, "--buy-size"),
    sample_ms: int = typer.Option(60_000, "--sample-ms", min=1),
    pad_seconds: int = typer.Option(120, "--pad-seconds", min=0),
) -> None:
    """
    Fetch window prices for one slug and run a **mid-price proxy** one-shot dual-limit simulation.

    Settlement uses Gamma outcome (Up => YES token pays 1); not order-accurate vs real CLOB.
    """
    slug_use = _slug_from_url_or_slug(market_url, slug)
    bundle = fetch_binary_window_prices(
        slug_use, sample_interval_ms=sample_ms, pad_seconds=pad_seconds
    )
    m = fetch_gamma_market_by_slug(slug_use)
    winner = resolved_winner_outcome_label(m) if m else None
    if winner == "Up":
        yes_at_1 = True
    elif winner == "Down":
        yes_at_1 = False
    else:
        yes_at_1 = True
        print(
            "[warn] Could not infer Up/Down settlement from Gamma; assuming YES token settles 1 (check manually)."
        )
    result = simulate_one_shot_dual_limit(
        bundle.aligned,
        buy_price=buy_price,
        sell_price=sell_price,
        size=buy_size,
        yes_expires_at_1=yes_at_1,
    )
    pprint(
        {
            "slug": bundle.slug,
            "settlement_up_wins": yes_at_1,
            "gamma_winner": winner,
            **result,
        }
    )


if __name__ == "__main__":
    app()
