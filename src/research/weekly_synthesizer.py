"""Build topic-first report inputs from LLM-classified research events."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from src.domain.news_taxonomy import DEFAULT_TOPIC, normalize_event_name, normalize_topic
from src.research.research_note_builder import resolve_event_display_text, sanitize_event_record
from src.utils.news_text import (
    coerce_bool,
    coerce_str,
    compute_char_ngram_similarity,
    normalize_whitespace,
)

_UNCERTAINTY_TERMS = ("预计", "预测", "指引", "假设", "传闻", "据称", "研报", "观点", "交流", "纪要", "目标价")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:万亿|万|亿|%|美元|元|gw|gwh|mw|eb|tb|gb|倍|吨|颗|个月|年)?", re.IGNORECASE)


def build_weekly_themes(
    events: list[dict[str, Any]],
    articles_by_url: dict[str, dict[str, Any]] | None = None,
    max_themes: int = 8,
) -> list[dict[str, Any]]:
    articles_by_url = articles_by_url or {}
    candidates = [_prepare_event(event, articles_by_url) for event in events]
    candidates = [event for event in candidates if _include_in_report(event)]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in candidates:
        groups[event["topic"]].append(event)

    themes = [_build_theme(topic, grouped) for topic, grouped in groups.items()]
    themes.sort(key=lambda item: (item["importance"], len(item["event_ids"]), item["theme"]), reverse=True)
    return themes[:max_themes]


def build_quality_flags(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = [_prepare_event(event, {}) for event in events]
    flags: list[dict[str, Any]] = []
    flags.extend(_numeric_flags(prepared))
    flags.extend(_confidence_flags(prepared))
    flags.extend(_duplicate_flags(prepared))
    return flags


def build_event_appendix(events: list[dict[str, Any]], limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
    rows = [_prepare_event(event, {}) for event in events]
    rows = [row for row in rows if _include_in_report(row)]
    rows.sort(key=lambda row: (-row["score"], row["topic"], row["event_name"]))
    if limit is not None:
        rows = rows[:limit]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["topic"]].append(_event_output(row))
    return dict(grouped)


def build_one_line_conclusion(themes: list[dict[str, Any]], appendix: dict[str, list[dict[str, Any]]]) -> str:
    if not themes:
        return "本期报告窗口内无足够有效事件形成明确主线。"
    top_topics = "、".join(theme["theme"] for theme in themes[:3])
    event_count = sum(len(rows) for rows in appendix.values())
    return f"本期主线集中在{top_topics}，共入选{event_count}条事件。"


def group_themes_by_topic(themes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for theme in themes:
        grouped[theme["topic"]].append(theme)
    return dict(grouped)


def group_themes_by_etf(themes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return group_themes_by_topic(themes)


def _prepare_event(event: dict[str, Any], articles_by_url: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sanitized = sanitize_event_record(event)
    topic = (
        normalize_topic(sanitized.get("topic"))
        or normalize_topic(sanitized.get("industry"))
        or DEFAULT_TOPIC
    )
    article = articles_by_url.get(coerce_str(sanitized.get("source_url")))
    summary = resolve_event_display_text(sanitized, article=article, max_chars=0)
    title = coerce_str(sanitized.get("title"))
    event_name = normalize_event_name(sanitized.get("event_name"), title)
    confidence = _coerce_float(sanitized.get("agent_confidence"), default=0.0)
    text = normalize_whitespace(" ".join([
        title,
        event_name,
        summary,
        coerce_str(sanitized.get("agent_reasoning")),
    ]))
    prepared = dict(sanitized)
    prepared.update({
        "topic": topic,
        "event_name": event_name,
        "title": title,
        "summary": summary,
        "source": coerce_str(sanitized.get("source_name")) or "未知来源",
        "source_url": coerce_str(sanitized.get("source_url")),
        "reasoning": coerce_str(sanitized.get("agent_reasoning")),
        "risk_or_caveat": coerce_str(sanitized.get("agent_risk_or_caveat")),
        "confidence": confidence,
        "text": text,
        "score": _score_event(sanitized, confidence, summary),
    })
    return prepared


def _include_in_report(event: dict[str, Any]) -> bool:
    if coerce_bool(event.get("is_noise")):
        return False
    return bool(event.get("event_id") or event.get("title") or event.get("event_name"))


def _score_event(event: dict[str, Any], confidence: float, summary: str) -> float:
    score = 2.0
    if confidence:
        score += confidence * 2
    tier = int(_coerce_float(event.get("source_tier"), default=0))
    if tier == 1:
        score += 1.2
    elif tier == 2:
        score += 1.0
    elif tier == 3:
        score += 0.5
    if summary:
        score += 0.8
    if coerce_str(event.get("agent_reasoning")):
        score += 0.4
    score += _recency_score(coerce_str(event.get("event_date")))
    return round(score, 3)


def _recency_score(raw_date: str) -> float:
    if not raw_date:
        return 0.0
    try:
        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
    return max(0.0, 0.8 - min(age_days, 7) * 0.08)


def _build_theme(topic: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: item["score"], reverse=True)
    key_facts = _dedupe_texts([event["summary"] for event in ordered], limit=4, max_chars=0)
    importance = min(
        10,
        max(1, int(round(
            sum(event["score"] for event in ordered) / max(1, len(ordered))
            + math.log2(len(ordered) + 1)
        ))),
    )
    event_outputs = [_event_output(event) for event in ordered]
    return {
        "theme": topic,
        "topic": topic,
        "direction": _infer_direction(ordered),
        "importance": importance,
        "summary": _theme_summary(topic, ordered),
        "key_facts": key_facts,
        "counterpoints": _dedupe_texts([
            event["risk_or_caveat"] for event in ordered if event.get("risk_or_caveat")
        ], limit=3, max_chars=0),
        "event_ids": [event["event_id"] for event in ordered if event.get("event_id")],
        "events": event_outputs,
        "titles": [event["event_name"] for event in ordered if event.get("event_name")],
        "sources": sorted({event["source"] for event in ordered if event.get("source")}),
        "quality_flags": [flag["message"] for flag in build_quality_flags(ordered)[:3]],
    }


def _event_output(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id", ""),
        "topic": event.get("topic", DEFAULT_TOPIC),
        "event_name": event.get("event_name") or event.get("title") or "未命名事件",
        "title": event.get("event_name") or event.get("title") or "未命名事件",
        "source": event.get("source", "未知来源"),
        "source_url": event.get("source_url", ""),
        "summary": event.get("summary", ""),
        "reasoning": event.get("reasoning", ""),
        "risk_or_caveat": event.get("risk_or_caveat", ""),
    }


def _theme_summary(topic: str, events: list[dict[str, Any]]) -> str:
    source_count = len({event["source"] for event in events if event.get("source")})
    event_count = len(events)
    top = events[0]
    reason = top.get("reasoning") or top.get("summary")
    return f"{topic}覆盖{event_count}条事件、{source_count}个来源。{reason}".strip()


def _infer_direction(events: list[dict[str, Any]]) -> str:
    text = " ".join(event["text"] for event in events)
    positive = sum(text.count(word) for word in ("利好", "上调", "涨价", "短缺", "放量", "超预期", "扩产", "突破"))
    negative = sum(text.count(word) for word in ("下调", "暴跌", "风险", "收缩", "不及", "过剩", "制裁", "压力"))
    if positive > negative:
        return "偏正面"
    if negative > positive:
        return "偏负面"
    return "中性/待验证"


def _numeric_flags(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    for event in events:
        numbers = _NUMBER_RE.findall(event["text"])
        has_uncertainty = any(term in event["text"] for term in _UNCERTAINTY_TERMS)
        if len(numbers) >= 6 and not has_uncertainty:
            flags.append(_flag(event, "数字密集", "摘要含大量数字但缺少预测/来源限制说明，建议核对原文。"))
        if any(_looks_extreme_number(token) for token in numbers) and not event.get("risk_or_caveat"):
            flags.append(_flag(event, "异常量级", "存在大额或高倍数数字且缺少 caveat，建议复核单位和来源。"))
    return flags


def _confidence_flags(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    for event in events:
        confidence = event.get("confidence", 0.0)
        if confidence and confidence < 0.55:
            flags.append(_flag(event, "低置信度", f"主题分类置信度 {confidence:.2f}，建议人工复核。"))
    return flags


def _duplicate_flags(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, left in enumerate(events):
        for right in events[i + 1:]:
            if (left["event_id"], right["event_id"]) in seen_pairs:
                continue
            seen_pairs.add((left["event_id"], right["event_id"]))
            if compute_char_ngram_similarity(left["event_name"], right["event_name"]) >= 0.72:
                flags.append(_flag(left, "疑似重复", f"与《{right['event_name']}》事件名高度相似。"))
                break
    return flags


def _looks_extreme_number(token: str) -> bool:
    text = token.lower()
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return False
    value = float(match.group(0))
    return "万亿" in text or value >= 10000 or ("倍" in text and value >= 10) or ("%" in text and value >= 100)


def _flag(event: dict[str, Any], kind: str, message: str) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id", ""),
        "source_url": event.get("source_url", ""),
        "title": event.get("event_name") or event.get("title", ""),
        "source": event.get("source", ""),
        "kind": kind,
        "message": message,
    }


def _dedupe_texts(texts: list[str], *, limit: int, max_chars: int) -> list[str]:
    result: list[str] = []
    for text in texts:
        normalized = normalize_whitespace(text)
        if not normalized:
            continue
        if any(compute_char_ngram_similarity(normalized, existing) >= 0.75 for existing in result):
            continue
        result.append(normalized if max_chars <= 0 else normalized[:max_chars].rstrip() + ("…" if len(normalized) > max_chars else ""))
        if len(result) >= limit:
            break
    return result


def _coerce_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result
