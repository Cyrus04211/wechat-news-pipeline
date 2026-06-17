"""Build human-readable article summaries and topic labels for reports."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.domain.news_taxonomy import normalize_topic
from src.sources.wechat_article_content import is_wechat_cdn_image_url, split_digest_and_cover
from src.utils.news_text import coerce_bool, coerce_list, coerce_str, normalize_whitespace

_MIN_MEANINGFUL_DIGEST_LEN = 8
_DIGIT_ONLY_RE = re.compile(r"^[\d\s.,]+$")


def is_meaningful_digest(text: str) -> bool:
    """Return True when MP digest looks like prose rather than cover URL or filler."""
    text = normalize_whitespace(text)
    if not text:
        return False
    if is_wechat_cdn_image_url(text):
        return False
    if _DIGIT_ONLY_RE.match(text):
        return False
    if len(text) < _MIN_MEANINGFUL_DIGEST_LEN:
        return False
    return True


def format_industry_label(industry: str) -> str:
    return normalize_topic(industry)


def classified_related_events(related_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep non-noise events related to an article."""
    return [
        event for event in related_events
        if not coerce_bool(event.get("is_noise"))
    ]


def collect_article_industries(
    article: dict[str, Any],
    related_events: list[dict[str, Any]],
) -> list[str]:
    industries: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        norm = normalize_topic(coerce_str(value))
        if norm and norm not in seen:
            seen.add(norm)
            industries.append(norm)

    for value in coerce_list(article.get("industry_candidates")):
        add(value)
    for value in coerce_list(article.get("tags")):
        add(value)
    for event in classified_related_events(related_events):
        add(event.get("topic") or event.get("industry"))
    return industries


def _dedupe_preserve_order(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        part = normalize_whitespace(part)
        if part and part not in seen:
            seen.add(part)
            ordered.append(part)
    return ordered


def parse_image_insights(raw_value: Any) -> list[dict[str, Any]]:
    """Parse persisted LLM image insight JSON into a normalized list."""
    if isinstance(raw_value, list):
        entries = raw_value
    elif not raw_value:
        return []
    else:
        try:
            entries = json.loads(str(raw_value))
        except json.JSONDecodeError:
            return []
    if not isinstance(entries, list):
        return []

    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            image_index = int(entry.get("image_index", 0))
        except (TypeError, ValueError):
            image_index = 0
        normalized.append({
            "image_index": image_index,
            "keep": bool(entry.get("keep", False)),
            "explanation": coerce_str(entry.get("explanation")),
        })
    return normalized


def kept_image_insights(raw_value: Any) -> list[dict[str, Any]]:
    return [entry for entry in parse_image_insights(raw_value) if entry.get("keep")]


def article_image_path_list(article: dict[str, Any]) -> list[str]:
    """Return ordered local image paths stored on an article record."""
    raw_paths = article.get("image_paths")
    if raw_paths is None:
        return []
    if isinstance(raw_paths, list):
        return [str(path) for path in raw_paths if path]
    return [str(raw_paths)] if raw_paths else []


def resolve_image_insight(
    article: dict[str, Any],
    *,
    local_path: str = "",
    image_index: int | None = None,
) -> dict[str, Any]:
    """Match a downloaded image file to its persisted LLM insight payload."""
    insights = parse_image_insights(article.get("agent_image_insights"))
    if not insights:
        return {}

    insight_map = {entry["image_index"]: entry for entry in insights}
    resolved_index = image_index

    if resolved_index is None and local_path:
        normalized = os.path.normpath(local_path)
        filename = Path(local_path).name
        for index, path in enumerate(article_image_path_list(article)):
            candidate = os.path.normpath(str(path))
            if candidate == normalized or Path(candidate).name == filename:
                resolved_index = index
                break

    if resolved_index is None:
        return {}

    return insight_map.get(resolved_index, {})


def build_article_display_summary(
    article: dict[str, Any],
    related_events: list[dict[str, Any]],
) -> str:
    """Prefer LLM article_summary, then fetched body, then classification reasoning."""
    agent_summaries = _dedupe_preserve_order([
        coerce_str(article.get("agent_summary")),
        *[
            coerce_str(event.get("agent_summary"))
            for event in classified_related_events(related_events)
            if coerce_str(event.get("agent_summary"))
        ],
    ])
    if agent_summaries:
        return "；".join(agent_summaries)

    content_text = normalize_whitespace(coerce_str(article.get("content_text")))
    if content_text:
        return content_text

    content_markdown = normalize_whitespace(coerce_str(article.get("content_markdown")))
    if content_markdown:
        return content_markdown

    reasonings = _dedupe_preserve_order([
        coerce_str(event.get("agent_reasoning"))
        for event in classified_related_events(related_events)
        if coerce_str(event.get("agent_reasoning"))
    ])
    if reasonings:
        return "；".join(reasonings)

    digest, _ = split_digest_and_cover(coerce_str(article.get("digest")))
    if is_meaningful_digest(digest):
        return digest

    title = coerce_str(article.get("title"))
    if title:
        return title

    return "暂无可用摘要（正文未抓取成功）"
