"""Build structured research notes and conclusions from classified events."""

from __future__ import annotations

from typing import Any, Optional

from src.domain.news_taxonomy import DEFAULT_TOPIC, normalize_topic
from src.research.article_summary import build_article_display_summary
from src.sources.wechat_article_content import normalize_wechat_article_url
from src.utils.news_text import coerce_optional_str, coerce_str, truncate_text


def index_articles_by_url(articles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map normalized article URLs to archive records for report fallbacks."""
    indexed: dict[str, dict[str, Any]] = {}
    for article in articles:
        url = normalize_wechat_article_url(coerce_str(article.get("source_url")))
        if url and url not in indexed:
            indexed[url] = article
    return indexed


def resolve_event_display_text(
    event: dict[str, Any],
    *,
    article: Optional[dict[str, Any]] = None,
    max_chars: int = 400,
) -> str:
    """Pick human-readable report text without leaking pandas NaN as ``nan``."""
    text = coerce_optional_str(
        event.get("agent_summary"),
        event.get("agent_reasoning"),
    )
    if not text and article:
        text = build_article_display_summary(article, [event])
    if not text:
        text = coerce_optional_str(
            event.get("summary"),
            event.get("title"),
        )
    return truncate_text(text, max_chars=max_chars)


def sanitize_event_record(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize pandas-loaded event dicts before templates or report builders."""
    sanitized = dict(event)
    for key in (
        "event_id",
        "title",
        "summary",
        "source_name",
        "source_url",
        "topic",
        "event_name",
        "industry",
        "agent_summary",
        "agent_reasoning",
        "event_date",
        "source_id",
    ):
        if key in sanitized:
            sanitized[key] = coerce_str(sanitized.get(key))
    return sanitized


def build_daily_conclusions(
    events: list[dict[str, Any]],
    *,
    articles_by_url: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, str]]:
    articles_by_url = articles_by_url or {}
    conclusions = []
    for raw_event in events:
        event = sanitize_event_record(raw_event)
        article = articles_by_url.get(normalize_wechat_article_url(event.get("source_url", "")))
        text = resolve_event_display_text(event, article=article)
        topic = (
            normalize_topic(event.get("topic"))
            or normalize_topic(event.get("industry"))
            or DEFAULT_TOPIC
        )
        conclusions.append({
            "text": text,
            "topic": topic,
            "industry": topic,
            "event_id": event.get("event_id", ""),
            "etf": "—",
            "source": event.get("source_name", "") or "未知来源",
        })
    if not conclusions:
        conclusions.append({
            "text": "本期报告窗口内无采集事件，请复核采集窗口或重新运行流水线。",
            "industry": "",
            "event_id": "",
            "etf": "",
            "source": "",
        })
    return conclusions


def build_industry_summary(events: list[dict[str, Any]]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    for raw_event in events:
        event = sanitize_event_record(raw_event)
        title = event.get("title", "")
        if not title:
            continue
        topic = (
            normalize_topic(event.get("topic"))
            or normalize_topic(event.get("industry"))
            or DEFAULT_TOPIC
        )
        summary.setdefault(topic, []).append(title)
    return summary


def build_watchlist(
    evidence_cards: list[dict[str, Any]],
    review_states: dict[str, dict],
) -> list[dict[str, Any]]:
    items = []
    for card in evidence_cards:
        eid = coerce_str(card.get("event_id", ""))
        status = review_states.get(eid, {}).get("status", card.get("reviewer_status", "new"))
        if status in {"watch", "new"}:
            items.append({
                "event_id": eid,
                "title": coerce_str(card.get("title", "")),
                "source_url": coerce_str(card.get("source_url", "")),
                "status": coerce_str(status),
                "topic": normalize_topic(card.get("topic")) or normalize_topic(card.get("industry")) or DEFAULT_TOPIC,
                "industry": normalize_topic(card.get("topic")) or normalize_topic(card.get("industry")) or DEFAULT_TOPIC,
                "verification_questions": card.get("verification_questions", []),
            })
    return items
