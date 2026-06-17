"""Download and index original images from WeChat article pages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests

from src.research.research_models import ArticleImage
from src.sources.wechat_article_content import (
    BROWSER_HEADERS,
    extract_image_urls_from_content_html,
    extract_wechat_image_urls_from_text,
    fetch_wechat_article_content,
    is_wechat_cdn_image_url,
)
from src.utils.news_storage import ensure_dir, write_records_csv
from src.utils.news_text import coerce_list, coerce_str

logger = logging.getLogger(__name__)

_IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|gif|webp)(?:\?|$)", re.I)


def _guess_extension(url: str, content_type: str = "") -> str:
    lower = (content_type or "").lower()
    if "png" in lower:
        return ".png"
    if "gif" in lower:
        return ".gif"
    if "webp" in lower:
        return ".webp"
    match = _IMAGE_EXT_RE.search(url)
    if match:
        return f".{match.group(1).lower().replace('jpeg', 'jpg')}"
    return ".jpg"


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        url = url.strip()
        if url and url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


class ArticleImageStore:
    def __init__(
        self,
        output_dir: str = "data/news/research/article_images",
        timeout: int = 30,
        rate_limit_seconds: float = 1.0,
    ):
        self.output_dir = output_dir
        self.timeout = timeout
        self.rate_limit_seconds = rate_limit_seconds
        self.index_path = os.path.join(output_dir, "image_index.jsonl")
        self.index_csv_path = os.path.join(output_dir, "image_index.csv")

    def load_index(self) -> list[ArticleImage]:
        if not os.path.exists(self.index_path):
            return []
        images: list[ArticleImage] = []
        with open(self.index_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    images.append(ArticleImage.from_dict(json.loads(line)))
        return images

    def save_index(self, images: list[ArticleImage]) -> None:
        ensure_dir(self.output_dir)
        stored = [img for img in images if img.local_path]
        with open(self.index_path, "w", encoding="utf-8") as handle:
            for image in stored:
                handle.write(json.dumps(image.to_dict(), ensure_ascii=False, default=str) + "\n")
        write_records_csv([img.to_dict() for img in stored], self.index_csv_path)

    def upsert_images(self, new_images: list[ArticleImage]) -> list[ArticleImage]:
        stored_new = [img for img in new_images if img.local_path]
        if not stored_new:
            return new_images
        existing = {img.image_id: img for img in self.load_index()}
        for image in stored_new:
            existing[image.image_id] = image
        merged = list(existing.values())
        self.save_index(merged)
        return stored_new

    def local_paths_for_article(self, article_id: str) -> list[str]:
        return [
            img.local_path
            for img in self.load_index()
            if img.article_id == article_id and img.local_path
        ]

    def images_for_article_ids(self, article_ids: set[str]) -> list[ArticleImage]:
        if not article_ids:
            return []
        return [
            img for img in self.load_index()
            if img.local_path and img.article_id in article_ids
        ]

    def images_for_events(self, event_ids: set[str]) -> list[ArticleImage]:
        if not event_ids:
            return []
        return [
            img for img in self.load_index()
            if img.local_path and any(eid in event_ids for eid in img.event_ids)
        ]

    def _collect_image_urls(
        self,
        *,
        content_html: str,
        digest: str,
        cover_image_url: str,
        source_url: str,
        session: Optional[requests.Session],
    ) -> tuple[list[str], str]:
        urls: list[str] = []
        fetch_status = "pending"

        if content_html:
            urls.extend(extract_image_urls_from_content_html(content_html))

        for candidate in (cover_image_url, digest):
            text = coerce_str(candidate)
            if is_wechat_cdn_image_url(text):
                urls.append(text)
            else:
                urls.extend(extract_wechat_image_urls_from_text(text))

        urls = _dedupe_urls(urls)
        if urls:
            return urls, "ok"
        if content_html:
            return [], "no_images"

        if source_url:
            result = fetch_wechat_article_content(source_url, timeout=self.timeout, session=session)
            if result.status == "ok":
                page_urls = list(result.image_urls or [])
                if result.content_html:
                    page_urls.extend(extract_image_urls_from_content_html(result.content_html))
                page_urls = _dedupe_urls(page_urls)
                if page_urls:
                    return page_urls, "ok"
                fetch_status = "no_images"
            else:
                fetch_status = result.status

        if fetch_status == "pending":
            fetch_status = "no_images"
        return [], fetch_status

    def fetch_and_store_for_article(
        self,
        *,
        article_id: str,
        source_url: str,
        related_event_ids: Optional[list[str]] = None,
        date_label: Optional[str] = None,
        content_html: str = "",
        digest: str = "",
        cover_image_url: str = "",
        session: Optional[requests.Session] = None,
    ) -> list[ArticleImage]:
        related_event_ids = coerce_list(related_event_ids)
        date_label = date_label or "unknown"

        image_urls, fetch_status = self._collect_image_urls(
            content_html=content_html,
            digest=digest,
            cover_image_url=cover_image_url,
            source_url=source_url,
            session=session,
        )

        if not image_urls:
            return [ArticleImage(
                image_id=hashlib.sha256(f"{article_id}:none".encode()).hexdigest()[:16],
                article_id=article_id,
                event_ids=related_event_ids,
                source_url=source_url,
                original_url="",
                local_path="",
                fetch_status=fetch_status,
                date_label=date_label,
            )]

        http = session or requests.Session()
        if session is None:
            http.headers.update(BROWSER_HEADERS)
            http.trust_env = False

        stored: list[ArticleImage] = []
        article_dir = ensure_dir(os.path.join(self.output_dir, date_label, article_id))
        last_fetch = 0.0

        for index, url in enumerate(image_urls):
            elapsed = time.time() - last_fetch
            if elapsed < self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds - elapsed)

            image_id = hashlib.sha256(f"{article_id}:{url}".encode()).hexdigest()[:16]
            local_path = ""
            status = "download_failed"

            try:
                resp = http.get(url, timeout=self.timeout, allow_redirects=True)
                resp.raise_for_status()
                ext = _guess_extension(url, resp.headers.get("Content-Type", ""))
                filename = f"{index:03d}_{image_id}{ext}"
                local_path = str(article_dir / filename)
                with open(local_path, "wb") as handle:
                    handle.write(resp.content)
                status = "ok"
            except Exception as exc:
                logger.warning("Failed to download image %s: %s", url, exc)

            last_fetch = time.time()
            stored.append(ArticleImage(
                image_id=image_id,
                article_id=article_id,
                event_ids=related_event_ids,
                source_url=source_url,
                original_url=url,
                local_path=local_path,
                fetch_status=status,
                date_label=date_label,
                index=index,
            ))

        return self.upsert_images(stored) or stored

    def build_for_articles(
        self,
        articles: list[dict[str, Any]],
        date_label: Optional[str] = None,
    ) -> list[ArticleImage]:
        all_images: list[ArticleImage] = []
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)
        session.trust_env = False

        for article in articles:
            published = coerce_str(article.get("published_at"))
            label = date_label or (published[:10] if published else "unknown")
            images = self.fetch_and_store_for_article(
                article_id=str(article.get("article_id", "")),
                source_url=str(article.get("source_url", "") or ""),
                related_event_ids=coerce_list(article.get("related_event_ids")),
                date_label=label,
                content_html=str(article.get("content_html", "") or ""),
                digest=str(article.get("digest", "") or ""),
                cover_image_url=str(article.get("cover_image_url", "") or ""),
                session=session,
            )
            all_images.extend(images)
        return all_images
