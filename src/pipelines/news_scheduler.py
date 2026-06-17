#!/usr/bin/env python3
from __future__ import annotations

"""
Scheduler daemon for recurring news collection.

Usage:
    python -m src.pipelines.news_scheduler --daemon --mode daily_brief
    python -m src.pipelines.news_scheduler --daemon --mode weekly_review
    python -m src.pipelines.news_scheduler --once --mode daily_brief
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.news_config import (
    load_collection_profiles_config,
    load_llm_config,
    load_runtime_config,
    load_schedule_config,
)
from src.pipelines.news_collection_profiles import COLLECTION_MODES, run_collection_profile

logger = logging.getLogger(__name__)
UTC = timezone.utc

DEFAULT_PROFILE_INTERVALS = {
    "daily_brief": 86400,
    "weekly_review": 604800,
}


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _default_state_path() -> str:
    runtime_config = load_runtime_config()
    schedule_config = load_schedule_config()
    return schedule_config.get(
        "state_path",
        os.path.join(runtime_config.get("log_dir", "data/news/logs"), "scheduler_state.json"),
    )


def load_schedule_state(state_path: str | None = None) -> dict:
    path = state_path or _default_state_path()
    if not os.path.exists(path):
        return {"profiles": {}}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle) or {"profiles": {}}
    except Exception as exc:
        logger.warning("Failed to load scheduler state from %s: %s", path, exc)
        return {"profiles": {}}


def save_schedule_state(state: dict, state_path: str | None = None) -> None:
    path = state_path or _default_state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def compute_profile_since(
    profile_name: str,
    *,
    profile_config: dict,
    state: dict,
    now: datetime | None = None,
) -> tuple[datetime, str]:
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    profile_state = (state.get("profiles") or {}).get(profile_name, {})
    last_success = _parse_timestamp(profile_state.get("last_success"))
    bootstrap_hours = int(profile_config.get("lookback_hours", 24))
    overlap_hours = float(profile_config.get("overlap_hours", 1))
    max_incremental_hours = int(profile_config.get("max_incremental_hours", bootstrap_hours))

    if last_success is None:
        return now_utc - timedelta(hours=bootstrap_hours), "bootstrap"

    incremental_since = last_success - timedelta(hours=overlap_hours)
    capped_since = now_utc - timedelta(hours=max_incremental_hours)
    if incremental_since < capped_since:
        return capped_since, "capped_incremental"
    return incremental_since, "incremental"


def run_profile_with_checkpoint(
    profile_name: str,
    *,
    state: dict,
    state_path: str,
    agent_provider: str,
    agent_api_key: str | None,
    agent_model: str,
    agent_base_url: str | None,
) -> int:
    profiles_config = load_collection_profiles_config()
    profile_config = profiles_config.get(profile_name, {})
    if not profile_config:
        raise ValueError(f"Unknown collection profile: {profile_name}")

    since, mode = compute_profile_since(profile_name, profile_config=profile_config, state=state)
    logger.info(
        "Running %s collection profile in %s mode since %s",
        profile_name,
        mode,
        since.isoformat(),
    )
    events = run_collection_profile(
        profile_name,
        agent_provider=agent_provider,
        agent_api_key=agent_api_key,
        agent_model=agent_model,
        agent_base_url=agent_base_url,
        since_override=since,
    )
    profiles_state = state.setdefault("profiles", {})
    profiles_state[profile_name] = {
        "last_success": datetime.now(UTC).isoformat(),
        "last_mode": mode,
        "last_retained_count": len(events),
        "last_since": since.isoformat(),
    }
    save_schedule_state(state, state_path=state_path)
    return len(events)


def daemon_loop(
    mode: str,
    *,
    agent_provider: str,
    agent_api_key: str | None,
    agent_model: str,
    agent_base_url: str | None,
) -> None:
    if mode not in COLLECTION_MODES:
        raise ValueError(f"Unknown scheduler mode: {mode}. Choose from {COLLECTION_MODES}.")

    logger.info("Starting scheduler daemon in %s mode", mode)
    schedule_config = load_schedule_config()
    state_path = schedule_config.get("state_path", _default_state_path())
    state = load_schedule_state(state_path)
    intervals = DEFAULT_PROFILE_INTERVALS.copy()
    intervals.update({
        str(name): int(seconds)
        for name, seconds in (schedule_config.get("profile_intervals_seconds", {}) or {}).items()
    })
    interval = intervals.get(mode, DEFAULT_PROFILE_INTERVALS.get(mode, 86400))
    last_run = 0.0

    while True:
        now = time.time()
        if not last_run or (now - last_run) >= interval:
            try:
                run_profile_with_checkpoint(
                    mode,
                    state=state,
                    state_path=state_path,
                    agent_provider=agent_provider,
                    agent_api_key=agent_api_key,
                    agent_model=agent_model,
                    agent_base_url=agent_base_url,
                )
                last_run = time.time()
            except Exception as exc:
                logger.error("%s collection failed: %s", mode, exc)

        sleep_time = max((last_run + interval) - time.time(), 10)
        logger.debug("Sleeping %.0fs until next %s collection", sleep_time, mode)
        time.sleep(sleep_time)


def once(
    mode: str,
    *,
    agent_provider: str,
    agent_api_key: str | None,
    agent_model: str,
    agent_base_url: str | None,
) -> None:
    if mode not in COLLECTION_MODES:
        raise ValueError(f"Unknown scheduler mode: {mode}. Choose from {COLLECTION_MODES}.")

    logger.info("Running one-time %s collection", mode)
    schedule_config = load_schedule_config()
    state_path = schedule_config.get("state_path", _default_state_path())
    state = load_schedule_state(state_path)
    run_profile_with_checkpoint(
        mode,
        state=state,
        state_path=state_path,
        agent_provider=agent_provider,
        agent_api_key=agent_api_key,
        agent_model=agent_model,
        agent_base_url=agent_base_url,
    )


def main():
    parser = argparse.ArgumentParser(description="News collection scheduler")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--mode",
        type=str,
        default="daily_brief",
        choices=list(COLLECTION_MODES),
        help="Collection mode: daily_brief (24h, once per day) or weekly_review (168h, once per week)",
    )
    parser.add_argument("--agent-provider", type=str, default=None,
                        choices=["auto", "closeai", "openai-compatible", "claude-code", "anthropic"],
                        help="Agent provider override")
    parser.add_argument("--agent-model", type=str, default=None, help="LLM model override")
    parser.add_argument("--agent-api-key", type=str, default=None, help="LLM API key override")
    parser.add_argument("--agent-base-url", type=str, default=None, help="LLM base URL override")
    args = parser.parse_args()

    log_dir = load_runtime_config().get("log_dir", "data/news/logs")
    llm_config = load_llm_config()
    agent_provider = args.agent_provider or llm_config.get("provider", "auto")
    agent_model = args.agent_model or llm_config.get("model", "deepseek-v4-pro")
    agent_base_url = args.agent_base_url or llm_config.get("api_base")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"scheduler_{datetime.now().strftime('%Y-%m-%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    if args.daemon:
        daemon_loop(
            args.mode,
            agent_provider=agent_provider,
            agent_api_key=args.agent_api_key,
            agent_model=agent_model,
            agent_base_url=agent_base_url,
        )
    else:
        once(
            args.mode,
            agent_provider=agent_provider,
            agent_api_key=args.agent_api_key,
            agent_model=agent_model,
            agent_base_url=agent_base_url,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
