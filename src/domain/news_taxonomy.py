from __future__ import annotations

from src.utils.news_text import coerce_str, normalize_whitespace

DEFAULT_TOPIC = "未分类"


def normalize_topic(value: object, *, max_chars: int = 40) -> str:
    """Normalize an LLM-generated topic label without forcing a fixed taxonomy."""
    text = normalize_whitespace(coerce_str(value))
    if not text:
        return ""
    if text.lower() in {"none", "null", "nan", "n/a"}:
        return ""
    if text in {"无", "无分类"}:
        return ""
    return text[:max_chars].rstrip()


def normalize_event_name(value: object, fallback: object = "", *, max_chars: int = 80) -> str:
    text = normalize_whitespace(coerce_str(value))
    if not text:
        text = normalize_whitespace(coerce_str(fallback))
    return text[:max_chars].rstrip()
