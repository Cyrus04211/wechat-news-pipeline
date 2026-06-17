"""Backfill missing LLM image insights for archived research articles."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import pandas as pd

from src.agents.image_insight_analyzer import ImageInsightAnalyzer
from src.agents.image_insight_utils import dump_image_insights, missing_image_insight_indices
from src.config.news_config import load_llm_config
from src.research.article_image_store import ArticleImageStore
from src.research.article_repository import ArticleRepository
from src.research.research_models import ResearchArticle
from src.utils.news_text import coerce_list, coerce_str

logger = logging.getLogger(__name__)


def ordered_image_paths_for_article(
    image_store: ArticleImageStore,
    article_id: str,
) -> list[str]:
    images = [
        image for image in image_store.load_index()
        if image.article_id == article_id and image.local_path
    ]
    images.sort(key=lambda item: (item.index, item.local_path))
    seen: set[str] = set()
    ordered: list[str] = []
    for image in images:
        path = os.path.normpath(image.local_path)
        if path in seen:
            continue
        seen.add(path)
        ordered.append(image.local_path)
    return ordered


class ImageInsightBackfill:
    def __init__(
        self,
        *,
        articles_dir: str,
        images_dir: str,
        llm_config: Optional[dict[str, Any]] = None,
        vision_batch_size: int = 4,
        sync_event_tables: bool = True,
        reviewed_path: str = "data/news/reviewed_all.csv",
        events_path: str = "data/news/events_classified.csv",
    ):
        self.repo = ArticleRepository(output_dir=articles_dir)
        self.image_store = ArticleImageStore(output_dir=images_dir)
        llm_config = llm_config or load_llm_config()
        self.analyzer = ImageInsightAnalyzer(
            llm_config=llm_config,
            vision_batch_size=vision_batch_size,
            vision_images_first=bool(llm_config.get("vision_images_first", True)),
            vision_disable_thinking=bool(llm_config.get("vision_disable_thinking", True)),
        )
        self.sync_event_tables = sync_event_tables
        self.reviewed_path = reviewed_path
        self.events_path = events_path

    def run(self, *, article_ids: Optional[set[str]] = None) -> dict[str, int]:
        stats = {"checked": 0, "updated": 0, "skipped": 0, "failed": 0, "pending": 0}
        articles = self.repo.load_articles()
        if not self.analyzer.enabled:
            for article in articles:
                if article_ids and article.article_id not in article_ids:
                    continue
                stats["checked"] += 1
                image_paths = ordered_image_paths_for_article(self.image_store, article.article_id)
                if not image_paths:
                    image_paths = list(article.image_paths or [])
                missing = missing_image_insight_indices(
                    image_paths,
                    article.agent_image_insights,
                )
                if missing and image_paths:
                    stats["pending"] += 1
            logger.warning(
                "Image insight backfill skipped: set CLOSEAI_API_KEY or OPENAI_API_KEY "
                "(%s articles still missing insights)",
                stats["pending"],
            )
            return stats
        changed_articles: list[ResearchArticle] = []
        article_updates: dict[str, str] = {}

        for article in articles:
            if article_ids and article.article_id not in article_ids:
                continue
            stats["checked"] += 1
            image_paths = ordered_image_paths_for_article(self.image_store, article.article_id)
            if not image_paths:
                image_paths = list(article.image_paths or [])
            if not image_paths:
                stats["skipped"] += 1
                continue

            missing = missing_image_insight_indices(
                image_paths,
                article.agent_image_insights,
            )
            if not missing:
                stats["skipped"] += 1
                continue

            industry = article.industry_candidates[0] if article.industry_candidates else ""
            try:
                merged = self.analyzer.analyze_missing_sync(
                    item_id=article.article_id,
                    title=article.title,
                    source_name=article.source_name,
                    content=article.content_text,
                    industry=industry,
                    reasoning="",
                    image_paths=image_paths,
                    existing_raw=article.agent_image_insights,
                )
            except Exception as exc:
                logger.warning("Backfill failed for %s: %s", article.article_id, exc)
                stats["failed"] += 1
                continue

            if not merged:
                stats["failed"] += 1
                continue

            payload = dump_image_insights(merged)
            if payload == article.agent_image_insights:
                stats["skipped"] += 1
                continue

            article.agent_image_insights = payload
            article.image_paths = image_paths
            changed_articles.append(article)
            article_updates[article.article_id] = payload
            stats["updated"] += 1

        if changed_articles:
            self.repo.upsert_articles(changed_articles)
            if self.sync_event_tables:
                self._sync_event_tables(article_updates)

        return stats

    def _sync_event_tables(self, article_updates: dict[str, str]) -> None:
        if not article_updates:
            return

        articles_by_id = {article.article_id: article for article in self.repo.load_articles()}
        for path in (self.reviewed_path, self.events_path):
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                logger.warning("Failed to load %s for insight sync: %s", path, exc)
                continue
            if "agent_image_insights" not in df.columns:
                df["agent_image_insights"] = "[]"

            updated_rows = 0
            for index, row in df.iterrows():
                source_url = coerce_str(row.get("source_url", ""))
                if not source_url:
                    continue
                article = self._match_article_for_row(row, articles_by_id)
                if not article or article.article_id not in article_updates:
                    continue
                df.at[index, "agent_image_insights"] = article_updates[article.article_id]
                image_paths = list(article.image_paths or [])
                if image_paths:
                    df.at[index, "selected_image_paths"] = json.dumps(image_paths, ensure_ascii=False)
                updated_rows += 1

            if updated_rows:
                df.to_csv(path, index=False)
                parquet_path = path.replace(".csv", ".parquet")
                if os.path.exists(parquet_path):
                    try:
                        df.to_parquet(parquet_path, index=False)
                    except Exception as exc:
                        logger.warning("Failed to update parquet %s: %s", parquet_path, exc)

    @staticmethod
    def _match_article_for_row(
        row: pd.Series,
        articles_by_id: dict[str, ResearchArticle],
    ) -> Optional[ResearchArticle]:
        for article in articles_by_id.values():
            if article.title and article.title == coerce_str(row.get("title", "")):
                return article
            event_ids = coerce_list(getattr(article, "related_event_ids", []))
            event_id = coerce_str(row.get("event_id", ""))
            if event_id and event_id in event_ids:
                return article
        return None
