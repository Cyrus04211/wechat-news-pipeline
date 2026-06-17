import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.agents.news_agent_orchestrator import AgentOrchestrator
from src.config.news_config import (
    load_llm_config,
    load_runtime_config,
    load_timeliness_config,
)
from src.collectors.article_media_prefetch import attach_article_media_to_events
from src.collectors.news_dedup import DeduplicationEngine, compute_dedup_group_id
from src.domain.news_taxonomy import DEFAULT_TOPIC
from src.sources.news_base import RawNewsItem
from src.sources.news_registry import build_source_adapters
from src.sources.news_wechat_mp import SessionExpiredError
from src.sources.wechat_article_content import extract_payload_metadata
from src.utils.news_storage import (
    ensure_dir,
    load_existing_dedup_keys,
    load_recent_titles,
    write_records_csv,
    write_records_parquet,
    write_events_csv,
    write_events_parquet,
    write_raw_jsonl,
)
from src.utils.news_source_health import SourceHealthTracker
from src.utils.news_text import (
    clean_summary_text,
    clean_title_text,
    coerce_bool,
    compute_event_id,
    is_multi_item_roundup,
    split_multi_item_roundup,
    is_transport_noise,
)
from src.utils.news_time import (
    configure_timeliness_windows,
    ensure_utc,
    set_timeliness_override,
    timeliness_decay,
    timeliness_window_hours,
    utc_now,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc


class ImpactNewsCollector:
    def __init__(
        self,
        industries: Optional[list[str]] = None,
        tiers: Optional[list[int]] = None,
        config_path: Optional[str] = None,
        debug: bool = False,
        use_agent: bool = False,
        agent_provider: str = "auto",
        agent_api_key: Optional[str] = None,
        agent_model: str = "deepseek-v4-pro",
        agent_base_url: Optional[str] = None,
        since: Optional[datetime] = None,
        source_ids: Optional[list[str]] = None,
        history_hours: Optional[int] = None,
    ):
        self.industries = industries or []
        self.tiers = tiers or [1, 2, 3, 4]
        self.config_path = config_path
        self.source_ids = source_ids
        self.runtime_config = load_runtime_config(config_path)
        self.llm_config = load_llm_config(config_path)
        self.timeliness_config = load_timeliness_config(config_path)
        self.raw_dir = self.runtime_config.get("raw_dir", "data/news/raw")
        self.output_parquet = self.runtime_config.get(
            "output_parquet", "data/news/events_classified.parquet"
        )
        self.output_csv = self.runtime_config.get(
            "output_csv", "data/news/events_classified.csv"
        )
        self.reviewed_output_parquet = self.runtime_config.get(
            "reviewed_output_parquet", "data/news/reviewed_all.parquet"
        )
        self.reviewed_output_csv = self.runtime_config.get(
            "reviewed_output_csv", "data/news/reviewed_all.csv"
        )
        self.dataset_id = self.runtime_config.get("dataset_id", "wechat_topic_classified_v1")
        self.debug = debug
        self.since = since
        self.history_hours = history_hours
        configure_timeliness_windows(self.timeliness_config)

        self.dedup = DeduplicationEngine(similarity_threshold=0.85)

        self.health_tracker = SourceHealthTracker(
            health_path=self.runtime_config.get("health_path", "data/news/logs/source_health.json"),
            max_consecutive_failures=self.runtime_config.get("max_consecutive_failures", 5),
            auto_disable=self.runtime_config.get("auto_disable_sources", True),
        )

        self.use_agent = use_agent
        self.agent = AgentOrchestrator(
            provider=agent_provider,
            api_key=agent_api_key,
            model=agent_model,
            base_url=agent_base_url or self.llm_config.get("api_base"),
            request_timeout_seconds=int(self.llm_config.get("request_timeout_seconds", 180)),
            concurrency=int(self.llm_config.get("concurrency", 4)),
            enabled=use_agent,
        )

        self.stats: dict[str, Any] = {
            "raw_items": 0,
            "transport_noise_dropped": 0,
            "roundup_split_items": 0,
            "roundup_dropped": 0,
            "timeliness_dropped": 0,
            "dedup_dropped": 0,
            "ingested": 0,
            "classified": 0,
            "llm_classified": 0,
            "llm_summarized": 0,
            "by_topic": Counter(),
            "by_source": Counter(),
            "errors": Counter(),
        }

    def _create_adapters(self) -> list:
        return build_source_adapters(
            industries=self.industries,
            tiers=self.tiers,
            config_path=self.config_path,
            source_ids=self.source_ids,
        )

    def collect_all(self) -> list[dict[str, Any]]:
        now = utc_now()
        adapters = self._create_adapters()

        if self.since:
            custom_hours = max(int((now - self.since).total_seconds() / 3600) + 1, 1)
            set_timeliness_override(custom_hours)

        existing_ids, existing_dedup_group_ids = load_existing_dedup_keys(self.output_parquet)
        self.dedup.load_existing(existing_ids, existing_dedup_group_ids)

        history_hours = self.history_hours
        if history_hours is None:
            history_hours = max(48, int((now - self.since).total_seconds() / 3600) + 1) if self.since else 48
        history = load_recent_titles(self.output_parquet, hours=history_hours)
        self.dedup.load_history(history)

        all_raw: list[RawNewsItem] = []

        disabled_sources = self.health_tracker.get_disabled_sources()
        if disabled_sources:
            logger.warning(f"Skipping {len(disabled_sources)} disabled sources: {', '.join(disabled_sources)}")

        for adapter in adapters:
            source_name = getattr(adapter, "_source_name", adapter.__class__.__name__)

            if self.health_tracker.is_disabled(source_name):
                logger.info(f"[{source_name}] Skipped (auto-disabled due to repeated failures)")
                self.stats["errors"][f"{source_name}(disabled)"] += 1
                continue

            try:
                tier = getattr(adapter, "_source_tier", 2)
                fetch_since = self.since if self.since else (now - timedelta(hours=timeliness_window_hours(tier)))
                items = adapter.fetch(since=fetch_since)
                all_raw.extend(items)
                self.health_tracker.record_success(source_name, len(items))
            except SessionExpiredError:
                logger.error(
                    "[%s] MP session expired – run 'python scripts/wechat_mp_login.py' to re-login",
                    source_name,
                )
                self.stats["errors"]["mp_session_expired"] += 1
                self.health_tracker.record_failure(source_name, "session expired")
                break
            except Exception as e:
                error_msg = str(e)[:200]
                logger.warning(f"[{source_name}] Fetch FAILED: {error_msg}")
                self.stats["errors"][source_name] += 1
                just_disabled = self.health_tracker.record_failure(source_name, error_msg)
                if just_disabled:
                    logger.warning(
                        f"[{source_name}] AUTO-DISABLED after "
                        f"{self.health_tracker.max_consecutive_failures} consecutive failures."
                    )

        self.stats["raw_items"] = len(all_raw)
        events = self._process_pipeline(all_raw, now)
        self._last_reviewed_events = events
        reviewed_events: list[dict[str, Any]] = []
        if self.use_agent and any(event.get("agent_classified", False) for event in events):
            reviewed_events = self._build_review_audit(events)
            self._write_review_audit(reviewed_events)

        classified_events = reviewed_events or [dict(event) for event in events]
        retained_events = [
            event for event in classified_events if event.get("retained_final", False)
        ] if reviewed_events else [
            dict(event)
            for event in events
            if self._final_decision_for_event(event)[0]
        ]
        events = classified_events
        self._last_events = events
        self.stats["retained"] = len(retained_events)
        self.stats["below_threshold"] = len(classified_events) - len(retained_events)
        self.stats["ingested"] = len(classified_events)
        self.stats["classified"] = len(classified_events)

        if events:
            ensure_dir(os.path.dirname(self.output_parquet))
            write_events_parquet(events, self.output_parquet)
            write_events_csv(events, self.output_csv, existing_path=self.output_csv)

            for event in events:
                if event.get("raw_payload"):
                    raw_payload = event["raw_payload"]
                    raw_items_list = [{
                        "title": event.get("title"),
                        "source_url": event.get("source_url"),
                        "event_id": event.get("event_id"),
                        "author": event.get("author"),
                        "raw_payload": raw_payload,
                    }] if isinstance(raw_payload, dict) else event["raw_payload"]
                    try:
                        event_date = event.get("event_date", now)
                        if isinstance(event_date, str):
                            from dateutil.parser import parse as dt_parse
                            event_date = dt_parse(event_date)
                        write_raw_jsonl(raw_items_list, event["source_name"], event_date, self.raw_dir)
                    except Exception as e:
                        logger.debug(f"Failed to write raw payload: {e}")

        set_timeliness_override(None)
        self._print_summary()
        return events

    def _build_review_audit(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reviewed = []
        for event in events:
            audited = dict(event)
            retained, drop_reason = self._final_decision_for_event(audited)
            audited["retained_final"] = retained
            audited["drop_reason"] = drop_reason
            reviewed.append(audited)
        return reviewed

    def _final_decision_for_event(self, event: dict[str, Any]) -> tuple[bool, str]:
        if coerce_bool(event.get("is_noise"), default=False):
            return False, "noise"
        if not (event.get("event_id") or event.get("title")):
            return False, "empty_event"
        return True, ""

    def _write_review_audit(self, reviewed_events: list[dict[str, Any]]) -> None:
        if not reviewed_events:
            return
        ensure_dir(os.path.dirname(self.reviewed_output_parquet))
        write_records_parquet(
            reviewed_events,
            self.reviewed_output_parquet,
            existing_path=self.reviewed_output_parquet,
        )
        write_records_csv(
            reviewed_events,
            self.reviewed_output_csv,
            existing_path=self.reviewed_output_csv,
        )

    def _create_raw_event(
        self,
        item: RawNewsItem,
        now: datetime,
        eid: str,
        event_date: datetime,
        dedup_group_id: str,
        timeliness_factor: float,
    ) -> dict[str, Any]:
        payload = item.raw_payload if isinstance(item.raw_payload, dict) else {}
        meta = extract_payload_metadata(payload)
        return {
            "event_id": eid,
            "event_date": event_date.isoformat(),
            "collected_time": now.isoformat(),
            "title": item.title,
            "summary": item.summary,
            "source_name": item.source_name,
            "source_url": item.source_url,
            "source_tier": item.source_tier,
            "author": meta["author"],
            "topic": "",
            "event_name": "",
            "industry": "",
            "is_noise": False,
            "agent_scored": False,
            "agent_classified": False,
            "agent_summarized": False,
            "agent_reasoning": "",
            "agent_confidence": 0.0,
            "agent_risk_or_caveat": "",
            "agent_summary": "",
            "agent_image_insights": "[]",
            "selected_image_paths": "[]",
            "article_content": "",
            "article_content_html": "",
            "article_content_status": "pending",
            "article_image_paths": [],
            "article_image_meta": "[]",
            "dedup_group_id": dedup_group_id,
            "dataset_id": self.dataset_id,
            "_timeliness_factor": timeliness_factor,
            "raw_payload": item.raw_payload if self.runtime_config.get("preserve_raw_payload", True) else None,
        }

    def _process_pipeline(
        self,
        raw_items: list[RawNewsItem],
        now: datetime,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        for item in raw_items:
            candidates = split_multi_item_roundup(item)
            if candidates:
                self.stats["roundup_split_items"] += len(candidates)
            elif is_multi_item_roundup(item.title, item.summary):
                self.stats["roundup_dropped"] += 1
                continue
            else:
                candidates = [item]

            for candidate in candidates:
                event_date = ensure_utc(candidate.event_date)
                title = clean_title_text(candidate.title)
                summary = clean_summary_text(candidate.summary)

                if is_transport_noise(title, summary):
                    self.stats["transport_noise_dropped"] += 1
                    continue

                timeliness_factor = timeliness_decay(event_date, now, candidate.source_tier)
                if timeliness_factor <= 0.0:
                    self.stats["timeliness_dropped"] += 1
                    continue

                eid = compute_event_id(title, candidate.source_name, event_date.isoformat(), "")
                dedup_group_id = compute_dedup_group_id(title, "", event_date)

                is_dup, _ = self.dedup.check(
                    title, candidate.source_name, event_date, "", dedup_group_id)
                if is_dup:
                    self.stats["dedup_dropped"] += 1
                    continue

                event = self._create_raw_event(
                    candidate, now, eid, event_date, dedup_group_id, timeliness_factor
                )
                event["title"] = title
                event["summary"] = summary
                events.append(event)

                self.stats["by_source"][candidate.source_name] += 1

                self.dedup.add_to_history({
                    "title": title, "industry": "", "event_date": event_date,
                    "dedup_group_id": dedup_group_id, "event_id": eid,
                })

        if self.use_agent and events:
            if not self.agent.can_classify_inline:
                raise RuntimeError(
                    "LLM topic classification is required but no inline API provider is available. "
                    "Configure CLOSEAI_API_KEY / OPENAI_API_KEY and llm.provider=closeai."
                )
            if self.runtime_config.get("fetch_article_content_for_llm", True):
                prefetch_stats = attach_article_media_to_events(
                    events,
                    timeout=int(self.runtime_config.get("article_fetch_timeout_seconds", 30)),
                    max_retries=int(self.runtime_config.get("article_fetch_max_retries", 2)),
                    rate_limit_seconds=float(
                        self.runtime_config.get("article_fetch_rate_limit_seconds", 2.0)
                    ),
                    max_chars=int(self.runtime_config.get("llm_article_content_max_chars", 0)),
                    max_images_per_event=int(self.runtime_config.get("llm_max_images_per_event", 0)),
                    media_cache_dir=str(
                        self.runtime_config.get("llm_media_cache_dir", "data/news/cache/llm_media")
                    ),
                    use_mp_cookies=bool(self.runtime_config.get("article_fetch_use_mp_cookies", True)),
                    playwright_fallback=bool(
                        self.runtime_config.get("article_fetch_playwright_fallback", False)
                    ),
                    html_cache_dir=str(
                        self.runtime_config.get("article_html_cache_dir", "data/news/cache/wechat_html")
                    ),
                )
                self.stats["article_prefetch"] = prefetch_stats

            events = self.agent.classify_items(
                events,
                batch_size=int(self.llm_config.get("batch_size", 12)),
                summary_concurrency=int(self.llm_config.get("summary_concurrency", 32)),
                vision_model=self.llm_config.get("vision_model") or self.llm_config.get("model"),
                vision_enabled=bool(self.llm_config.get("vision_enabled", True)),
                vision_batch_size=int(self.llm_config.get("vision_batch_size", 4)),
                vision_images_first=bool(self.llm_config.get("vision_images_first", True)),
                vision_disable_thinking=bool(self.llm_config.get("vision_disable_thinking", True)),
            )
            self.stats["llm_classified"] = sum(1 for e in events if e.get("agent_classified", False))
            self.stats["llm_summarized"] = sum(1 for e in events if e.get("agent_summarized", False))

        for e in events:
            e.pop("_timeliness_factor", None)
            e.pop("article_content", None)
            e.pop("article_content_html", None)
            e.pop("article_content_status", None)
            e.pop("article_image_paths", None)
            e.pop("article_image_meta", None)
            topic = e.get("topic", "") or DEFAULT_TOPIC
            self.stats["by_topic"][topic] += 1

        return events

    def _print_summary(self) -> None:
        lines = [
            f"raw_items={self.stats['raw_items']}",
            f"transport_noise_dropped={self.stats['transport_noise_dropped']}",
            f"roundup_split_items={self.stats['roundup_split_items']}",
            f"roundup_dropped={self.stats['roundup_dropped']}",
            f"timeliness_dropped={self.stats['timeliness_dropped']}",
            f"dedup_dropped={self.stats['dedup_dropped']}",
            f"ingested={self.stats['ingested']}",
            f"llm_classified={self.stats['llm_classified']}",
            f"llm_summarized={self.stats['llm_summarized']}",
            f"by_topic={dict(self.stats['by_topic'])}",
            f"by_source={dict(self.stats['by_source'])}",
            f"errors={dict(self.stats['errors'])}",
        ]
        if self.use_agent:
            lines.append(f"agent={self.agent.get_summary()}")

        classified = self.stats.get("classified", 0)
        retained = self.stats.get("retained", 0)
        if classified:
            lines.append(
                f"classified_all={classified} retained_for_report={retained} "
                f"(non-noise, audit only)"
            )
        raw = self.stats["raw_items"]
        if raw > 0 and classified:
            classify_rate = classified / raw * 100
            lines.append(f"classify_rate={classify_rate:.1f}% ({classified}/{raw} llm-processed)")

        disabled = self.health_tracker.get_disabled_sources()
        if disabled:
            lines.append(f"disabled_sources={disabled}")

        summary = "\n".join(lines)
        logger.info(f"\n{summary}")
        print(f"\n=== Collection Summary ===\n{summary}\n")

        if disabled:
            print(self.health_tracker.get_health_report())
