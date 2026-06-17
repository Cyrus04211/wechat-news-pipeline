from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from src.collectors.news_impact import ImpactNewsCollector
from src.config.news_config import (
    load_collection_profiles_config,
    load_llm_config,
)
from src.utils.news_time import utc_now

logger = logging.getLogger(__name__)

COLLECTION_MODES = ("daily_brief", "weekly_review")


def resolve_collection_lookback_hours(
    mode: str,
    config_path: Optional[str] = None,
) -> int:
    profiles = load_collection_profiles_config(config_path)
    profile_config = profiles.get(mode, {})
    if not profile_config:
        raise ValueError(f"Unknown collection mode: {mode}. Choose from {COLLECTION_MODES}.")
    return int(profile_config.get("lookback_hours", 24))


def lookback_hours_to_report_days(lookback_hours: int) -> int:
    return max(1, (int(lookback_hours) + 23) // 24)


def resolve_report_plan(
    *,
    collection_mode: str,
    report_type: Optional[str] = None,
    days: Optional[int] = None,
    config_path: Optional[str] = None,
) -> tuple[str, int, int]:
    """Align report period with the configured collection lookback window."""
    lookback_hours = resolve_collection_lookback_hours(collection_mode, config_path=config_path)
    resolved_days = days if days is not None else lookback_hours_to_report_days(lookback_hours)
    resolved_type = report_type or collection_mode
    return resolved_type, resolved_days, lookback_hours


def resolve_profile_sequence(
    mode: str,
    config_path: Optional[str] = None,
) -> list[str]:
    profiles = load_collection_profiles_config(config_path)
    if mode in profiles:
        return [mode]
    return []


def run_collection_profile(
    profile_name: str,
    *,
    industries: Optional[list[str]] = None,
    debug: bool = False,
    agent_provider: Optional[str] = None,
    agent_api_key: Optional[str] = None,
    agent_model: Optional[str] = None,
    agent_base_url: Optional[str] = None,
    llm_batch_size: Optional[int] = None,
    config_path: Optional[str] = None,
    since_override: Optional[datetime] = None,
) -> list[dict]:
    profile_config = load_collection_profiles_config(config_path).get(profile_name, {})
    if not profile_config:
        raise ValueError(f"Unknown collection profile: {profile_name}")

    llm_config = load_llm_config(config_path)
    lookback_hours = int(profile_config.get("lookback_hours", 24))
    history_hours = int(profile_config.get("history_hours", max(lookback_hours, 48)))
    profile_industries = profile_config.get("industries") or industries
    profile_tiers = profile_config.get("tiers")
    profile_source_ids = profile_config.get("source_ids")
    since = since_override or (utc_now() - timedelta(hours=lookback_hours))

    logger.info(
        "Running collection profile %s: lookback=%sh sources=%s",
        profile_name,
        lookback_hours,
        ",".join(profile_source_ids or []) or "all",
    )

    collector = ImpactNewsCollector(
        industries=profile_industries,
        tiers=profile_tiers,
        debug=debug,
        use_agent=True,
        agent_provider=agent_provider or llm_config.get("provider", "auto"),
        agent_api_key=agent_api_key,
        agent_model=agent_model or llm_config.get("model", "deepseek-v4-pro"),
        agent_base_url=agent_base_url or llm_config.get("api_base"),
        since=since,
        source_ids=profile_source_ids,
        history_hours=history_hours,
        config_path=config_path,
    )
    collector.llm_config["batch_size"] = (
        llm_batch_size
        if llm_batch_size is not None
        else int(profile_config.get("llm_batch_size", llm_config.get("batch_size", 12)))
    )
    return collector.collect_all()


def run_collection_mode(
    mode: str,
    *,
    industries: Optional[list[str]] = None,
    debug: bool = False,
    agent_provider: Optional[str] = None,
    agent_api_key: Optional[str] = None,
    agent_model: Optional[str] = None,
    agent_base_url: Optional[str] = None,
    llm_batch_size: Optional[int] = None,
    config_path: Optional[str] = None,
) -> dict[str, int]:
    profile_name = mode
    events = run_collection_profile(
        profile_name,
        industries=industries,
        debug=debug,
        agent_provider=agent_provider,
        agent_api_key=agent_api_key,
        agent_model=agent_model,
        agent_base_url=agent_base_url,
        llm_batch_size=llm_batch_size,
        config_path=config_path,
    )
    return {profile_name: len(events)}
