"""Article archive and manual import for research workflows."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.domain.news_taxonomy import normalize_topic
from src.research.article_summary import parse_image_insights
from src.research.research_models import ResearchArticle
from src.sources.wechat_article_content import (
    build_wechat_http_session,
    extract_payload_metadata,
    fetch_wechat_article_content,
    is_valid_wechat_article_url,
    normalize_wechat_article_url,
    split_digest_and_cover,
)
from src.utils.news_storage import ensure_dir, write_records_csv, write_records_parquet

logger = logging.getLogger(__name__)
UTC = timezone.utc


def compute_content_hash(
    title: str,
    source_name: str,
    digest: str = "",
    content: str = "",
    source_url: str = "",
) -> str:
    payload = "|".join([title or "", source_name or "", digest or "", content or "", source_url or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_article_id(source_url: str = "", title: str = "", content_hash: str = "") -> str:
    seed = f"{source_url}|{title}|{content_hash}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def article_fetch_kwargs_from_runtime(
    runtime_config: dict | None = None,
    research_config: dict | None = None,
) -> dict[str, Any]:
    """Map runtime configs to ``build_from_pipeline`` fetch args."""
    runtime_config = runtime_config or {}
    research_config = research_config or {}
    research = research_config.get("research", research_config)
    return {
        "fetch_content": bool(runtime_config.get("fetch_article_content", True)),
        "fetch_images": bool(research.get("fetch_article_images", True)),
        "content_fetch_timeout": int(runtime_config.get("article_fetch_timeout_seconds", 30)),
        "content_fetch_max_retries": int(runtime_config.get("article_fetch_max_retries", 2)),
        "content_fetch_rate_limit": float(runtime_config.get("article_fetch_rate_limit_seconds", 2.0)),
        "use_mp_cookies": bool(runtime_config.get("article_fetch_use_mp_cookies", True)),
        "playwright_fallback": bool(runtime_config.get("article_fetch_playwright_fallback", False)),
        "html_cache_dir": str(runtime_config.get("article_html_cache_dir", "data/news/cache/wechat_html")),
        "images_output_dir": str(
            research.get("article_images_dir", "data/news/research/article_images")
        ),
    }


def _parse_json_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value.strip() else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    return []


class ArticleRepository:
    def __init__(self, output_dir: str = "data/news/research/articles"):
        self.output_dir = output_dir
        self.parquet_path = os.path.join(output_dir, "articles.parquet")
        self.csv_path = os.path.join(output_dir, "articles.csv")
        self.jsonl_path = os.path.join(output_dir, "articles.jsonl")

    def load_articles(self) -> list[ResearchArticle]:
        if os.path.exists(self.parquet_path):
            try:
                df = pd.read_parquet(self.parquet_path)
                return [ResearchArticle.from_dict(row) for row in df.to_dict(orient="records")]
            except Exception as exc:
                logger.warning(f"Failed to load articles parquet: {exc}")
        if os.path.exists(self.jsonl_path):
            articles = []
            with open(self.jsonl_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        articles.append(ResearchArticle.from_dict(json.loads(line)))
            return articles
        return []

    def save_articles(self, articles: list[ResearchArticle]) -> dict[str, str]:
        ensure_dir(self.output_dir)
        records = [a.to_dict() for a in articles]
        pd.DataFrame(records).to_parquet(self.parquet_path, index=False)
        write_records_csv(records, self.csv_path)
        with open(self.jsonl_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return {
            "parquet": self.parquet_path,
            "csv": self.csv_path,
            "jsonl": self.jsonl_path,
        }

    @staticmethod
    def _merge_article_updates(old: ResearchArticle, article: ResearchArticle) -> None:
        old.related_event_ids = sorted(set(old.related_event_ids + article.related_event_ids))
        if article.industry_candidates:
            old.industry_candidates = sorted(set(old.industry_candidates + article.industry_candidates))
        if article.notes and not old.notes:
            old.notes = article.notes
        if article.content_text and not old.content_text:
            old.content_text = article.content_text
        if article.content_markdown and not old.content_markdown:
            old.content_markdown = article.content_markdown
        if article.content_html and not old.content_html:
            old.content_html = article.content_html
        if article.cover_image_url and not old.cover_image_url:
            old.cover_image_url = article.cover_image_url
        if article.digest and not old.digest:
            old.digest = article.digest
        if article.agent_summary and (
            not old.agent_summary or len(article.agent_summary) > len(old.agent_summary)
        ):
            old.agent_summary = article.agent_summary
        if article.agent_image_insights:
            old_count = len({
                entry["image_index"]
                for entry in parse_image_insights(old.agent_image_insights)
                if str(entry.get("explanation", "") or "").strip()
            })
            new_count = len({
                entry["image_index"]
                for entry in parse_image_insights(article.agent_image_insights)
                if str(entry.get("explanation", "") or "").strip()
            })
            if new_count >= old_count:
                old.agent_image_insights = article.agent_image_insights
        if article.image_paths:
            old.image_paths = sorted(set(old.image_paths + article.image_paths))
        if article.images_fetch_status != "pending":
            if article.image_paths or old.images_fetch_status in ("pending", ""):
                old.images_fetch_status = article.images_fetch_status

    def upsert_articles(self, new_articles: list[ResearchArticle]) -> list[ResearchArticle]:
        existing = {a.article_id: a for a in self.load_articles()}
        hash_index = {a.content_hash: a for a in existing.values() if a.content_hash}
        merged: list[ResearchArticle] = list(existing.values())

        for article in new_articles:
            if article.content_hash and article.content_hash in hash_index:
                self._merge_article_updates(hash_index[article.content_hash], article)
                continue
            if article.article_id in existing:
                self._merge_article_updates(existing[article.article_id], article)
                continue
            merged.append(article)
            existing[article.article_id] = article
            if article.content_hash:
                hash_index[article.content_hash] = article

        self.save_articles(merged)
        return merged

    def query(
        self,
        article_id: Optional[str] = None,
        source_name: Optional[str] = None,
        date: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> list[ResearchArticle]:
        results = self.load_articles()
        if article_id:
            results = [a for a in results if a.article_id == article_id]
        if source_name:
            results = [a for a in results if source_name in a.source_name]
        if date:
            results = [a for a in results if (a.published_at or "").startswith(date)]
        if industry:
            norm = normalize_topic(industry)
            results = [
                a for a in results
                if norm in a.industry_candidates or norm in a.tags
            ]
        return results

    def build_from_pipeline(
        self,
        reviewed_path: str = "data/news/reviewed_all.csv",
        events_path: str = "data/news/events_classified.csv",
        raw_dir: str = "data/news/raw",
        fetch_content: bool = False,
        fetch_images: bool = False,
        content_fetch_timeout: int = 30,
        content_fetch_max_retries: int = 2,
        content_fetch_rate_limit: float = 2.0,
        use_mp_cookies: bool = True,
        playwright_fallback: bool = False,
        html_cache_dir: str = "data/news/cache/wechat_html",
        images_output_dir: str = "data/news/research/article_images",
    ) -> list[ResearchArticle]:
        articles: dict[str, ResearchArticle] = {}
        sources = []

        for path in (reviewed_path, events_path):
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
                sources.append(df)
            except Exception as exc:
                logger.warning(f"Failed to read {path}: {exc}")

        if not sources:
            return []

        df = pd.concat(sources, ignore_index=True)
        if "event_id" in df.columns:
            df = df.drop_duplicates(subset=["event_id"], keep="last")

        for _, row in df.iterrows():
            try:
                article = self._event_row_to_article(row.to_dict(), raw_dir)
                if article.article_id not in articles:
                    articles[article.article_id] = article
                else:
                    existing = articles[article.article_id]
                    eid = str(row.get("event_id", ""))
                    if eid and eid not in existing.related_event_ids:
                        existing.related_event_ids.append(eid)
                    ind = normalize_topic(row.get("topic")) or normalize_topic(row.get("industry", ""))
                    if ind and ind not in existing.industry_candidates:
                        existing.industry_candidates.append(ind)
                    row_summary = str(row.get("agent_summary", "") or "").strip()
                    if row_summary and (
                        not existing.agent_summary
                        or len(row_summary) > len(existing.agent_summary)
                    ):
                        existing.agent_summary = row_summary
            except Exception as exc:
                logger.warning(f"Skip article from row: {exc}")

        self._enrich_from_raw(articles, raw_dir)
        self._backfill_industries_from_events(articles, df)
        article_list = list(articles.values())
        if fetch_content:
            self._fetch_missing_content(
                article_list,
                timeout=content_fetch_timeout,
                max_retries=content_fetch_max_retries,
                rate_limit=content_fetch_rate_limit,
                use_mp_cookies=use_mp_cookies,
                playwright_fallback=playwright_fallback,
                html_cache_dir=html_cache_dir,
            )
        if fetch_images:
            self._fetch_article_images(article_list, images_output_dir)
        return self.upsert_articles(article_list)

    def _event_row_to_article(self, row: dict[str, Any], raw_dir: str) -> ResearchArticle:
        title = str(row.get("title", "") or "")
        source_name = str(row.get("source_name", "") or "")
        source_url = str(row.get("source_url", "") or "")
        digest = str(row.get("summary", "") or row.get("digest", "") or "")
        event_id = str(row.get("event_id", "") or "")
        published = str(row.get("event_date", "") or row.get("published_at", "") or "")
        collected = str(row.get("collected_time", "") or "")
        industry = normalize_topic(row.get("topic")) or normalize_topic(row.get("industry", ""))
        author = str(row.get("author", "") or "")
        raw_payload = row.get("raw_payload")
        if isinstance(raw_payload, str) and raw_payload.strip():
            try:
                raw_payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                raw_payload = None
        if isinstance(raw_payload, dict):
            meta = extract_payload_metadata(raw_payload)
            if meta["author"] and not author:
                author = meta["author"]
            if meta["digest"] and not digest:
                digest = meta["digest"]

        digest, cover_image_url = split_digest_and_cover(digest)
        agent_summary = str(row.get("agent_summary", "") or "").strip()
        agent_image_insights = str(row.get("agent_image_insights", "") or "[]").strip() or "[]"
        selected_paths = _parse_json_string_list(row.get("selected_image_paths"))
        content_hash = compute_content_hash(title, source_name, digest or cover_image_url, "", source_url)
        article_id = make_article_id(source_url, title, content_hash)

        fetch_status = "success"
        content_text = ""
        content_markdown = ""
        raw_path = self._find_raw_path(raw_dir, source_name, published)

        if not digest and not title:
            fetch_status = "failed"
        elif not content_text and not content_markdown:
            fetch_status = "partial"

        return ResearchArticle(
            article_id=article_id,
            source_id=str(row.get("source_id", "") or source_name),
            source_name=source_name,
            source_tier=int(row.get("source_tier", 0) or 0),
            title=title,
            author=author,
            published_at=published,
            collected_at=collected,
            source_url=source_url,
            industry_candidates=[industry] if industry else [],
            digest=digest,
            agent_summary=agent_summary,
            agent_image_insights=agent_image_insights,
            cover_image_url=cover_image_url,
            content_text=content_text,
            content_markdown=content_markdown,
            content_hash=content_hash,
            fetch_status=fetch_status,
            image_paths=selected_paths,
            images_fetch_status="ok" if selected_paths else "pending",
            raw_path=raw_path,
            related_event_ids=[event_id] if event_id else [],
            tags=[industry] if industry else [],
        )

    def _find_raw_path(self, raw_dir: str, source_name: str, published_at: str) -> str:
        if not source_name or not os.path.isdir(raw_dir):
            return ""
        date_part = ""
        if published_at:
            date_part = published_at[:10]
        source_dir = os.path.join(raw_dir, source_name)
        if not os.path.isdir(source_dir):
            return ""
        if date_part:
            candidate = os.path.join(source_dir, f"{date_part}.jsonl")
            if os.path.exists(candidate):
                return candidate
        files = sorted(Path(source_dir).glob("*.jsonl"), reverse=True)
        return str(files[0]) if files else ""

    def _enrich_from_raw(self, articles: dict[str, ResearchArticle], raw_dir: str) -> None:
        if not os.path.isdir(raw_dir):
            return
        for article in articles.values():
            if not article.raw_path or not os.path.exists(article.raw_path):
                continue
            try:
                with open(article.raw_path, encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        if not self._raw_record_matches_article(record, article):
                            continue

                        payload = record.get("raw_payload") or record
                        if isinstance(payload, str):
                            try:
                                payload = json.loads(payload.replace("'", '"'))
                            except (json.JSONDecodeError, ValueError):
                                payload = {}
                        if not isinstance(payload, dict):
                            break

                        meta = extract_payload_metadata(payload)
                        raw_digest = meta["digest"] or article.digest
                        digest, cover = split_digest_and_cover(raw_digest)
                        if digest:
                            article.digest = digest
                        if cover and not article.cover_image_url:
                            article.cover_image_url = cover
                        elif meta.get("cover") and not article.cover_image_url:
                            article.cover_image_url = meta["cover"]
                        if meta["author"] and not article.author:
                            article.author = meta["author"]
                        break
            except Exception:
                pass

    @staticmethod
    def _backfill_industries_from_events(
        articles: dict[str, ResearchArticle],
        df: pd.DataFrame,
    ) -> None:
        if df.empty or "event_id" not in df.columns:
            return
        event_industry: dict[str, str] = {}
        for row in df.to_dict(orient="records"):
            eid = str(row.get("event_id", "") or "")
            ind = normalize_topic(row.get("topic")) or normalize_topic(row.get("industry", ""))
            if eid and ind:
                event_industry[eid] = ind

        for article in articles.values():
            for eid in article.related_event_ids:
                ind = event_industry.get(eid)
                if ind and ind not in article.industry_candidates:
                    article.industry_candidates.append(ind)
                if ind and ind not in article.tags:
                    article.tags.append(ind)
            article.industry_candidates = sorted(set(article.industry_candidates))
            article.tags = sorted(set(article.tags))

    @staticmethod
    def _raw_record_matches_article(record: dict[str, Any], article: ResearchArticle) -> bool:
        record_url = normalize_wechat_article_url(str(record.get("source_url", "") or ""))
        article_url = normalize_wechat_article_url(article.source_url or "")
        if record_url and article_url and record_url == article_url:
            return True

        record_title = str(record.get("title", "") or "").strip()
        if record_title and record_title == article.title:
            return True

        payload = record.get("raw_payload") if isinstance(record.get("raw_payload"), dict) else record
        if isinstance(payload, dict):
            meta = extract_payload_metadata(payload)
            if meta["link"] and article_url and meta["link"] == article_url:
                return True
            if meta["title"] and meta["title"] == article.title:
                return True
        return False

    def _fetch_missing_content(
        self,
        articles: list[ResearchArticle],
        *,
        timeout: int = 30,
        max_retries: int = 2,
        rate_limit: float = 2.0,
        use_mp_cookies: bool = True,
        playwright_fallback: bool = False,
        html_cache_dir: str = "",
    ) -> None:
        """Fetch full article body for archive entries that only have digest."""
        last_fetch = 0.0
        http = build_wechat_http_session(use_mp_cookies=use_mp_cookies)
        for article in articles:
            if article.content_text or article.fetch_status == "manual_import":
                continue
            if not article.source_url or not is_valid_wechat_article_url(article.source_url):
                continue

            elapsed = time.time() - last_fetch
            if elapsed < rate_limit:
                time.sleep(rate_limit - elapsed)

            result = fetch_wechat_article_content(
                article.source_url,
                timeout=timeout,
                max_retries=max_retries,
                session=http,
                use_mp_cookies=use_mp_cookies,
                playwright_fallback=playwright_fallback,
                cache_dir=html_cache_dir or None,
            )
            last_fetch = time.time()

            if result.status != "ok":
                if article.fetch_status == "success":
                    article.fetch_status = "partial"
                continue

            article.content_text = result.content_text
            article.content_html = result.content_html
            article.content_markdown = result.content_text
            if result.author and not article.author:
                article.author = result.author
            if result.title and not article.title:
                article.title = result.title
            article.content_hash = compute_content_hash(
                article.title,
                article.source_name,
                article.digest,
                article.content_text,
                article.source_url,
            )
            article.fetch_status = "success"

    def _fetch_article_images(
        self,
        articles: list[ResearchArticle],
        images_output_dir: str,
    ) -> None:
        from src.research.article_image_store import ArticleImageStore

        store = ArticleImageStore(output_dir=images_output_dir)
        for article in articles:
            if not article.source_url:
                article.images_fetch_status = "no_url"
                continue
            published = str(article.published_at or "")[:10] or "unknown"
            if article.image_paths:
                article.image_paths = self._materialize_image_paths(
                    article.image_paths,
                    images_output_dir,
                    article.article_id,
                    published,
                )
                article.images_fetch_status = "ok" if article.image_paths else "no_images"
                continue
            stored = store.fetch_and_store_for_article(
                article_id=article.article_id,
                source_url=article.source_url,
                related_event_ids=article.related_event_ids,
                date_label=published,
                content_html=article.content_html,
                digest=article.digest,
                cover_image_url=article.cover_image_url,
            )
            paths = [img.local_path for img in stored if img.local_path]
            article.image_paths = paths
            if paths:
                article.images_fetch_status = "ok"
            else:
                article.images_fetch_status = stored[0].fetch_status if stored else "no_images"

    @staticmethod
    def _materialize_image_paths(
        paths: list[str],
        images_output_dir: str,
        article_id: str,
        date_label: str,
    ) -> list[str]:
        """Copy LLM-selected cache images into the archive image directory."""
        if not paths:
            return []
        dest_dir = ensure_dir(os.path.join(images_output_dir, date_label, article_id))
        materialized: list[str] = []
        for index, src in enumerate(paths):
            src_path = Path(src)
            if not src_path.exists():
                continue
            dest = dest_dir / f"{index:03d}_{src_path.name}"
            if not dest.exists():
                shutil.copy2(src_path, dest)
            materialized.append(str(dest))
        return materialized

    def import_manual(
        self,
        file_path: str,
        source_name: str = "手动导入",
        industry: str = "",
        url: str = "",
        tags: Optional[list[str]] = None,
    ) -> ResearchArticle:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        suffix = path.suffix.lower()
        raw = path.read_text(encoding="utf-8", errors="replace")
        content_text = ""
        content_markdown = ""

        if suffix in {".md", ".markdown"}:
            content_markdown = raw
            content_text = re.sub(r"[#*_`\[\]]", "", raw)
        elif suffix == ".html":
            content_text = re.sub(r"<[^>]+>", " ", raw)
            content_markdown = raw
        else:
            content_text = raw

        title = ""
        if content_markdown.startswith("#"):
            title = content_markdown.split("\n", 1)[0].lstrip("# ").strip()
        if not title:
            title = path.stem

        norm_industry = normalize_topic(industry)
        now = datetime.now(UTC).isoformat()
        content_hash = compute_content_hash(title, source_name, "", content_text, url)
        article_id = make_article_id(url or str(path), title, content_hash)

        article = ResearchArticle(
            article_id=article_id,
            source_id=source_name,
            source_name=source_name,
            source_tier=0,
            title=title,
            published_at=now,
            collected_at=now,
            source_url=url,
            industry_candidates=[norm_industry] if norm_industry else [],
            digest=content_text,
            content_text=content_text,
            content_markdown=content_markdown,
            content_hash=content_hash,
            fetch_status="manual_import",
            raw_path=str(path.resolve()),
            tags=(tags or []) + ([norm_industry] if norm_industry else []),
            notes=f"manual_import:{path.name}",
        )
        self.upsert_articles([article])
        return article


def industry_to_etf(industry: str) -> str:
    return ""
