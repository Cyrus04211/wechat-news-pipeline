"""Fetch WeChat article bodies and images before LLM classification."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

from src.sources.wechat_article_content import (
    build_wechat_http_session,
    download_wechat_image,
    extract_payload_metadata,
    extract_wechat_image_urls_from_text,
    fetch_wechat_article_content,
    is_valid_wechat_article_url,
    is_wechat_cdn_image_url,
    split_digest_and_cover,
)
from src.utils.news_storage import ensure_dir
from src.utils.news_text import coerce_str, normalize_whitespace

logger = logging.getLogger(__name__)


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        url = (url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


def _collect_image_urls_for_event(event: dict[str, Any], content_html: str, image_urls: list[str]) -> list[str]:
    urls = list(image_urls or [])
    if content_html:
        from src.sources.wechat_article_content import extract_image_urls_from_content_html

        urls.extend(extract_image_urls_from_content_html(content_html))

    payload = event.get("raw_payload")
    if isinstance(payload, dict):
        meta = extract_payload_metadata(payload)
        digest, cover = split_digest_and_cover(meta.get("digest", ""))
        if cover:
            urls.append(cover)
        elif meta.get("cover"):
            urls.append(meta["cover"])
        if is_wechat_cdn_image_url(coerce_str(event.get("summary"))):
            urls.append(coerce_str(event.get("summary")))
        urls.extend(extract_wechat_image_urls_from_text(digest))

    return _dedupe_urls(urls)


def attach_article_media_to_events(
    events: list[dict[str, Any]],
    *,
    timeout: int = 30,
    max_retries: int = 2,
    rate_limit_seconds: float = 2.0,
    max_chars: int = 0,
    max_images_per_event: int = 0,
    media_cache_dir: str = "data/news/cache/llm_media",
    use_mp_cookies: bool = True,
    playwright_fallback: bool = False,
    html_cache_dir: str = "data/news/cache/wechat_html",
) -> dict[str, int]:
    """Fetch article text/HTML and download images for multimodal LLM classification."""
    stats = {
        "attempted": 0,
        "ok": 0,
        "failed": 0,
        "skipped": 0,
        "images_downloaded": 0,
    }
    last_fetch = 0.0
    http = build_wechat_http_session(use_mp_cookies=use_mp_cookies)

    for event in events:
        event.setdefault("article_content", "")
        event.setdefault("article_content_html", "")
        event.setdefault("article_content_status", "pending")
        event.setdefault("article_image_paths", [])
        event.setdefault("article_image_meta", "[]")
        event.setdefault("agent_summary", "")
        event.setdefault("agent_image_insights", "[]")
        event.setdefault("selected_image_paths", "[]")

        url = coerce_str(event.get("source_url"))
        if not url or not is_valid_wechat_article_url(url):
            event["article_content_status"] = "no_url"
            stats["skipped"] += 1
            continue

        if coerce_str(event.get("article_content")) and event.get("article_image_paths"):
            event["article_content_status"] = "cached"
            stats["skipped"] += 1
            continue

        elapsed = time.time() - last_fetch
        if elapsed < rate_limit_seconds:
            time.sleep(rate_limit_seconds - elapsed)

        stats["attempted"] += 1
        result = fetch_wechat_article_content(
            url,
            timeout=timeout,
            max_retries=max_retries,
            session=http,
            use_mp_cookies=use_mp_cookies,
            playwright_fallback=playwright_fallback,
            cache_dir=html_cache_dir,
        )
        last_fetch = time.time()

        if result.status != "ok" or not result.content_text:
            event["article_content_status"] = result.status
            stats["failed"] += 1
            logger.debug("Article media prefetch failed for %s: %s", url, result.status)
            continue

        text = normalize_whitespace(result.content_text)
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        event["article_content"] = text
        event["article_content_html"] = result.content_html
        event["article_content_status"] = "ok"

        image_urls = _collect_image_urls_for_event(event, result.content_html, result.image_urls)
        if max_images_per_event and max_images_per_event > 0:
            image_urls = image_urls[:max_images_per_event]

        event_id = coerce_str(event.get("event_id")) or hashlib.sha256(url.encode()).hexdigest()[:16]
        image_dir = ensure_dir(os.path.join(media_cache_dir, event_id))
        image_meta: list[dict[str, Any]] = []
        local_paths: list[str] = []

        for index, image_url in enumerate(image_urls):
            ext = "jpg"
            if "wx_fmt=" in image_url:
                ext = image_url.split("wx_fmt=", 1)[1].split("&", 1)[0] or "jpg"
            filename = f"{index:03d}_{hashlib.sha256(image_url.encode()).hexdigest()[:8]}.{ext}"
            dest = os.path.join(image_dir, filename)
            if download_wechat_image(image_url, dest, session=http, timeout=timeout):
                local_paths.append(dest)
                image_meta.append({
                    "index": index,
                    "local_path": dest,
                    "original_url": image_url,
                })
                stats["images_downloaded"] += 1

        event["article_image_paths"] = local_paths
        event["article_image_meta"] = json.dumps(image_meta, ensure_ascii=False)
        stats["ok"] += 1

    return stats


# Backward-compatible alias used by older tests/imports.
attach_article_content_to_events = attach_article_media_to_events
