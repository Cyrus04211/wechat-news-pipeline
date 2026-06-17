from __future__ import annotations

import inspect
import logging
from typing import Callable, Optional

from src.config.news_config import load_sources_config
from src.sources.news_base import NewsSourceAdapter
from src.sources.news_wechat_mp import WeChatMPAdapter, WeChatMPSessionManager

logger = logging.getLogger(__name__)


_REGISTRY: dict[str, Callable[..., NewsSourceAdapter]] = {
    "wechat_mp": WeChatMPAdapter,
}


def available_source_ids() -> list[str]:
    return list(_REGISTRY)


def _build_adapter_kwargs(
    factory: Callable[..., NewsSourceAdapter],
    config: dict,
) -> dict:
    kwargs = {}
    params = inspect.signature(factory).parameters
    if "rate_limit_seconds" in config:
        kwargs["rate_limit"] = config["rate_limit_seconds"]
    if "timeout_seconds" in config:
        kwargs["timeout"] = config["timeout_seconds"]
    if "timeout" in config:
        kwargs["timeout"] = config["timeout"]
    if "max_retries" in config:
        kwargs["max_retries"] = config["max_retries"]
    if "page_delay_seconds" in config:
        kwargs["page_delay"] = config["page_delay_seconds"]
    if "account" in config:
        kwargs["account_name"] = str(config["account"])
    if "account_name" in config:
        kwargs["account_name"] = str(config["account_name"])
    if config.get("fakeid"):
        kwargs["fakeid"] = str(config["fakeid"])
    return kwargs


def build_source_adapters(
    industries: Optional[list[str]] = None,
    tiers: Optional[list[int]] = None,
    config_path: Optional[str] = None,
    source_ids: Optional[list[str]] = None,
) -> list[NewsSourceAdapter]:
    source_config = load_sources_config(config_path)
    adapters: list[NewsSourceAdapter] = []
    allowed_source_ids = set(source_ids or [])

    # Create a shared MP session manager if any source uses wechat_mp type
    mp_session: Optional[WeChatMPSessionManager] = None
    has_mp = any(
        c.get("type") == "wechat_mp" and c.get("enabled", False)
        for c in source_config.values()
    )
    if has_mp:
        mp_session = WeChatMPSessionManager()
        mp_session.load()
        status, detail = mp_session.probe_status()
        if status in {"missing_token", "expired", "invalid"}:
            logger.warning(
                "MP session is invalid or expired. "
                "Run 'python scripts/wechat_mp_login.py' to re-login."
            )
        elif status == "unreachable":
            logger.warning(
                "Could not verify MP session via searchbiz (%s). "
                "Continuing with saved session; collection may still work.",
                detail,
            )

    for source_id, config in source_config.items():
        if not config.get("enabled", False):
            continue
        if allowed_source_ids and source_id not in allowed_source_ids:
            continue

        adapter_type = config.get("type", "wechat_mp")
        factory = _REGISTRY.get(adapter_type)
        if factory is None:
            continue

        source_tier = int(config.get("tier", 0) or 0)
        if tiers and source_tier not in tiers:
            continue

        kwargs = _build_adapter_kwargs(factory, config)
        if adapter_type == "wechat_mp" and mp_session is not None:
            kwargs["session_manager"] = mp_session
        adapter = factory(**kwargs)

        # Set metadata on the adapter instance
        adapter._source_name = config.get("name", source_id)
        adapter._source_tier = source_tier

        adapters.append(adapter)

    return adapters
