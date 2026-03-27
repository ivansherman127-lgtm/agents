"""
File-based stop signals for long-running strategies.

The dashboard (or shell) writes markers under ``data/control/``; the trader loop
polls :func:`stop_pending`. Run the bot from the repository root (default), or
paths resolve from this package location.
"""

from __future__ import annotations

from pathlib import Path

from agents.application.session_metrics import safe_session_file_stem

REPO_ROOT = Path(__file__).resolve().parents[2]


def control_dir() -> Path:
    d = REPO_ROOT / "data" / "control"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stop_all_path() -> Path:
    return control_dir() / "STOP_ALL"


def stop_session_path(session_id: str) -> Path:
    return control_dir() / f"stop_{safe_session_file_stem(session_id)}"


def stop_slug_path(market_slug: str) -> Path:
    slug = market_slug.strip()
    if not slug:
        return control_dir() / "stop_slug__empty"
    return control_dir() / f"stop_slug_{safe_session_file_stem(slug)}"


def stop_pending(session_id: str, market_slug: str = "") -> bool:
    if stop_all_path().is_file():
        return True
    if stop_session_path(session_id).is_file():
        return True
    if market_slug.strip() and stop_slug_path(market_slug).is_file():
        return True
    return False


def request_stop_all() -> None:
    stop_all_path().touch()


def request_stop_session(session_id: str) -> None:
    stop_session_path(session_id).touch()


def request_stop_slug(market_slug: str) -> None:
    if market_slug.strip():
        stop_slug_path(market_slug).touch()


def clear_stop_all() -> None:
    p = stop_all_path()
    if p.is_file():
        p.unlink()


def clear_session_stop_files(session_id: str, market_slug: str = "") -> None:
    """Remove per-session / per-slug markers (not ``STOP_ALL``)."""
    p = stop_session_path(session_id)
    if p.is_file():
        p.unlink()
    if market_slug.strip():
        p2 = stop_slug_path(market_slug)
        if p2.is_file():
            p2.unlink()


def list_stop_marker_names() -> list[str]:
    if not control_dir().exists():
        return []
    return sorted(p.name for p in control_dir().iterdir() if p.is_file())
