import hashlib
import html
import math
import re
import unicodedata
from typing import Any, Optional


def coerce_str(value: Any) -> str:
    """Coerce CSV/parquet field to str; treat None/NaN as empty."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def coerce_optional_str(*values: Any) -> str:
    """Return the first non-empty string after :func:`coerce_str`."""
    for value in values:
        text = coerce_str(value)
        if text:
            return text
    return ""


def truncate_text(text: str, max_chars: int = 400) -> str:
    """Trim long prose for report bullets; append ellipsis when truncated."""
    text = coerce_str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return default


def coerce_list(value: Any) -> list[str]:
    """Coerce CSV/parquet list fields; avoid numpy truth-value errors."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            value = value.tolist()
        except Exception:
            return []
    if isinstance(value, list):
        return [
            str(v) for v in value
            if v is not None and not (isinstance(v, float) and math.isnan(v)) and str(v).strip()
        ]
    if isinstance(value, str) and value.strip():
        if value.startswith("["):
            import json
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return coerce_list(parsed)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def normalize_title_for_dedup(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^\w\s]", "", text)
    prefixes = r"^(快讯|突发|独家|更新|update|breaking)\s*"
    text = re.sub(prefixes, "", text, flags=re.IGNORECASE).strip()
    suffixes = r"\s*(快讯|来源.*)$"
    text = re.sub(suffixes, "", text).strip()
    return text


def compute_title_similarity(a: str, b: str) -> float:
    na = normalize_title_for_dedup(a)
    nb = normalize_title_for_dedup(b)
    if na == nb:
        return 1.0
    set_a = set(na.split())
    set_b = set(nb.split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def compute_char_ngram_similarity(a: str, b: str, n: int = 3) -> float:
    na = normalize_title_for_dedup(a)
    nb = normalize_title_for_dedup(b)
    if na == nb:
        return 1.0
    if len(na) < n or len(nb) < n:
        return compute_title_similarity(a, b)

    ngrams_a = {na[i:i+n] for i in range(len(na) - n + 1)}
    ngrams_b = {nb[i:i+n] for i in range(len(nb) - n + 1)}
    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union)


def compute_event_id(title: str, source_name: str, event_date: str, industry: str) -> str:
    raw = f"{normalize_title_for_dedup(title)}|{source_name}|{event_date}|{industry}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def normalize_whitespace(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def clean_title_text(title: str) -> str:
    cleaned = normalize_whitespace(title)
    cleaned = cleaned.strip(" -|")
    return cleaned


def clean_summary_text(summary: Optional[str]) -> Optional[str]:
    cleaned = normalize_whitespace(summary or "")
    return cleaned or None


def is_multi_item_roundup(title: str, summary: Optional[str] = None) -> bool:
    headline = clean_title_text(title)
    body = clean_summary_text(summary or "") or ""
    if not headline:
        return False

    has_roundup_title = bool(re.search(r"(早报|午报|晚报|周报|盘点|汇总|速览|简报)", headline))
    if not has_roundup_title:
        return False

    numbered_items = len(re.findall(r"(?:^|\s)\d{1,2}[、.．)]", body))
    detail_links = body.count("查看详情")
    has_digest_intro = "今天的重要" in body or "现在是" in body or "咱们明天见" in body
    return numbered_items >= 3 or detail_links >= 3 or has_digest_intro


def split_multi_item_roundup(item) -> list:
    title = clean_title_text(getattr(item, "title", ""))
    summary = getattr(item, "summary", None)
    if not is_multi_item_roundup(title, summary):
        return []

    raw_payload = getattr(item, "raw_payload", {}) or {}
    summary_html = ""
    if isinstance(raw_payload, dict):
        summary_detail = raw_payload.get("summary_detail")
        if isinstance(summary_detail, dict):
            summary_html = str(summary_detail.get("value", "") or "")
        if not summary_html:
            summary_html = str(raw_payload.get("summary", "") or raw_payload.get("description", "") or "")
    if not summary_html:
        return []

    section_pattern = re.compile(r"<h2[^>]*>(.*?)</h2>\s*(.*?)(?=<h2[^>]*>|$)", re.IGNORECASE | re.DOTALL)
    sections = section_pattern.findall(summary_html)
    if len(sections) < 2:
        return []

    from src.sources.news_base import RawNewsItem

    children = []
    for index, (heading_html, body_html) in enumerate(sections, start=1):
        child_title = clean_title_text(normalize_whitespace(heading_html))
        child_title = re.sub(r"^\d{1,2}[、.．)]\s*", "", child_title).strip()
        if not child_title:
            continue

        child_summary = clean_summary_text(body_html) or ""
        child_summary = re.sub(r"\s*>>\s*查看详情.*$", "", child_summary).strip()
        child_summary = re.sub(r"\s*查看详情.*$", "", child_summary).strip()

        link_match = re.search(r'href=["\']([^"\']+)["\']', body_html, re.IGNORECASE)
        child_url = html.unescape(link_match.group(1)).strip() if link_match else (getattr(item, "source_url", "") or "")

        children.append(RawNewsItem(
            source_name=item.source_name,
            source_tier=item.source_tier,
            title=child_title,
            event_date=item.event_date,
            source_url=child_url,
            summary=child_summary or None,
            source_item_id=f"{getattr(item, 'source_item_id', '') or getattr(item, 'source_url', '')}#roundup-{index}",
            raw_payload={
                "split_from_roundup": True,
                "roundup_parent_title": item.title,
                "roundup_parent_url": item.source_url,
                "roundup_index": index,
                "child_title": child_title,
                "child_url": child_url,
                "child_summary": child_summary,
            },
        ))

    return children


def is_transport_noise(title: str, summary: Optional[str] = None) -> bool:
    headline = clean_title_text(title)
    if not headline:
        return True

    if re.fullmatch(r"[\d\s:：,，./-]+", headline):
        return True

    date_patterns = [
        r"^\d{1,2}月\d{1,2}日[，,\s]*星期[一二三四五六日天][，,\s]*\d{1,2}:\d{2}(:\d{2})?$",
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(\s+\d{1,2}:\d{2}(:\d{2})?)?$",
        r"^\d{1,2}:\d{2}(:\d{2})?$",
    ]
    if any(re.fullmatch(pattern, headline) for pattern in date_patterns):
        return True

    junk_markers = {
        "加载中",
        "查看更多",
        "点击查看",
        "返回顶部",
        "打开app",
        "下载app",
        "登录",
        "注册",
        "分享",
        "评论",
        "收藏",
    }
    if headline.lower() in junk_markers:
        return True

    if len(headline) < 8 and not (summary or "").strip():
        return True

    return False
