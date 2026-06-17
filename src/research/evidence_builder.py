"""Build evidence cards from classified events and article index."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Optional

import pandas as pd

from src.domain.news_taxonomy import DEFAULT_TOPIC, normalize_topic
from src.research.article_repository import ArticleRepository
from src.research.research_models import EvidenceCard, ResearchArticle
from src.research.review_workflow import ReviewWorkflow
from src.utils.news_storage import ensure_dir, write_records_csv, write_records_parquet
from src.utils.news_text import coerce_str
logger = logging.getLogger(__name__)

DEFAULT_VERIFICATION_QUESTIONS = [
    "该事件是否有第二来源验证？",
    "是否有价格、销量、订单、政策原文、公司公告支撑？",
    "是否能归入当前大话题下的核心事件？",
]

DEFAULT_FOLLOW_UP_ACTIONS = [
    "查原始政策文件",
    "查商品价格数据",
    "查公司公告",
    "查同主题其他来源",
]


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.strip():
        if value.startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if v]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _extract_key_claims(row: dict[str, Any]) -> list[str]:
    title = coerce_str(row.get("title", ""))
    reasoning = coerce_str(row.get("agent_reasoning", ""))
    summary = coerce_str(row.get("summary", ""))
    result = []
    if title:
        result.append(title)
    if reasoning and reasoning not in result:
        result.append(reasoning)
    elif summary and summary not in result:
        result.append(summary)
    return result


def _extract_supporting_evidence(row: dict[str, Any], article: Optional[ResearchArticle]) -> list[str]:
    from src.research.article_summary import build_article_display_summary, is_meaningful_digest
    from src.sources.wechat_article_content import is_wechat_cdn_image_url
    result = []
    title = coerce_str(row.get("title", ""))
    if title:
        result.append(f"标题：{title}")

    reasoning = coerce_str(row.get("agent_reasoning", ""))
    if reasoning:
        result.append(f"分类理由：{reasoning}")

    row_summary = coerce_str(row.get("summary", ""))
    if row_summary and not is_wechat_cdn_image_url(row_summary) and is_meaningful_digest(row_summary):
        result.append(f"公众号 digest：{row_summary}")

    if article:
        article_dict = article.to_dict()
        display_summary = build_article_display_summary(article_dict, [row])
        if display_summary and display_summary not in {title, reasoning, row_summary}:
            if article.content_text:
                result.append(f"正文：{display_summary}")
            else:
                result.append(f"摘要：{display_summary}")
    return result


def compute_evidence_quality_score(row: dict[str, Any]) -> float:
    score = 0.0
    if coerce_str(row.get("source_url", "")):
        score += 3
    if coerce_str(row.get("event_date", "")) or coerce_str(row.get("published_at", "")):
        score += 2
    if coerce_str(row.get("agent_reasoning", "")):
        score += 2
    tier = int(row.get("source_tier", 0) or 0)
    if tier <= 2 and tier > 0:
        score += 2
    elif tier == 3:
        score += 1
    return min(round(score, 1), 10.0)


def _find_article_for_event(
    row: dict[str, Any],
    articles: list[ResearchArticle],
) -> Optional[ResearchArticle]:
    event_id = coerce_str(row.get("event_id", ""))
    source_url = coerce_str(row.get("source_url", ""))
    title = coerce_str(row.get("title", ""))
    source_name = coerce_str(row.get("source_name", ""))

    for article in articles:
        if event_id and event_id in article.related_event_ids:
            return article
    for article in articles:
        if source_url and article.source_url and source_url == article.source_url:
            return article
    for article in articles:
        if title and article.title == title and source_name == article.source_name:
            return article
    return None


def _should_include_event(record: dict[str, Any]) -> bool:
    """Include every collected event that was written to the review/classified store."""
    event_id = str(record.get("event_id", "") or "").strip()
    title = str(record.get("title", "") or "").strip()
    return bool(event_id or title)


class EvidenceBuilder:
    def __init__(
        self,
        output_dir: str = "data/news/research/evidence_cards",
        review_workflow: Optional[ReviewWorkflow] = None,
    ):
        self.output_dir = output_dir
        self.parquet_path = os.path.join(output_dir, "evidence_cards.parquet")
        self.csv_path = os.path.join(output_dir, "evidence_cards.csv")
        self.jsonl_path = os.path.join(output_dir, "evidence_cards.jsonl")
        self.review_workflow = review_workflow or ReviewWorkflow()

    def load_events(
        self,
        reviewed_path: str = "data/news/reviewed_all.csv",
        events_path: str = "data/news/events_classified.csv",
    ) -> pd.DataFrame:
        frames = []
        for path in (reviewed_path, events_path):
            if os.path.exists(path):
                try:
                    frames.append(pd.read_csv(path))
                except Exception as exc:
                    logger.warning(f"Failed to read {path}: {exc}")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if "event_id" in df.columns:
            df = df.drop_duplicates(subset=["event_id"], keep="last")
        return df

    def build(
        self,
        reviewed_path: str = "data/news/reviewed_all.csv",
        events_path: str = "data/news/events_classified.csv",
        articles: Optional[list[ResearchArticle]] = None,
        article_repo: Optional[ArticleRepository] = None,
    ) -> list[EvidenceCard]:
        df = self.load_events(reviewed_path, events_path)
        if articles is None:
            repo = article_repo or ArticleRepository()
            articles = repo.load_articles()

        review_states = self.review_workflow.load_states()
        cards: list[EvidenceCard] = []

        for _, row in df.iterrows():
            record = row.to_dict()
            if not _should_include_event(record):
                continue
            cards.append(self._row_to_card(record, articles, review_states))

        return self.save_cards(cards, review_states)

    def _row_to_card(
        self,
        row: dict[str, Any],
        articles: list[ResearchArticle],
        review_states: dict[str, dict],
    ) -> EvidenceCard:
        event_id = coerce_str(row.get("event_id", ""))
        article = _find_article_for_event(row, articles)
        topic = (
            normalize_topic(row.get("topic"))
            or normalize_topic(row.get("industry"))
            or DEFAULT_TOPIC
        )
        verification = _parse_list(row.get("verification_questions"))
        follow_up = _parse_list(row.get("follow_up_actions"))

        if not verification:
            verification = list(DEFAULT_VERIFICATION_QUESTIONS)
        if not follow_up:
            follow_up = list(DEFAULT_FOLLOW_UP_ACTIONS)

        review = review_states.get(event_id, {})
        reviewer_status = review.get("status", "new")
        reviewer_notes = review.get("note", "")
        evidence_id = hashlib.sha256(f"evidence:{event_id}".encode()).hexdigest()[:16]

        return EvidenceCard(
            evidence_id=evidence_id,
            event_id=event_id,
            article_id=article.article_id if article else "",
            title=coerce_str(row.get("title", "")),
            source_name=coerce_str(row.get("source_name", "")),
            source_url=coerce_str(row.get("source_url", "")),
            published_at=coerce_str(row.get("event_date", "")),
            industry=topic,
            etf="",
            key_claims=_extract_key_claims(row),
            supporting_evidence=_extract_supporting_evidence(row, article),
            evidence_quality_score=compute_evidence_quality_score(row),
            verification_questions=verification,
            follow_up_actions=follow_up,
            reviewer_status=reviewer_status,
            reviewer_notes=reviewer_notes,
        )

    def save_cards(
        self,
        cards: list[EvidenceCard],
        review_states: Optional[dict[str, dict]] = None,
    ) -> list[EvidenceCard]:
        ensure_dir(self.output_dir)
        existing_notes: dict[str, str] = {}
        if os.path.exists(self.parquet_path):
            try:
                old_df = pd.read_parquet(self.parquet_path)
                for _, row in old_df.iterrows():
                    eid = str(row.get("event_id", ""))
                    note = str(row.get("reviewer_notes", "") or "")
                    if eid and note:
                        existing_notes[eid] = note
            except Exception:
                pass

        review_states = review_states or self.review_workflow.load_states()
        for card in cards:
            if card.event_id in review_states:
                card.reviewer_status = review_states[card.event_id].get("status", card.reviewer_status)
                saved_note = review_states[card.event_id].get("note", "")
                if saved_note:
                    card.reviewer_notes = saved_note
            elif card.event_id in existing_notes and not card.reviewer_notes:
                card.reviewer_notes = existing_notes[card.event_id]

        records = [c.to_dict() for c in cards]
        write_records_parquet(records, self.parquet_path)
        write_records_csv(records, self.csv_path)
        with open(self.jsonl_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return cards

    def load_cards(self) -> list[EvidenceCard]:
        if os.path.exists(self.parquet_path):
            try:
                df = pd.read_parquet(self.parquet_path)
                return [EvidenceCard.from_dict(row) for row in df.to_dict(orient="records")]
            except Exception:
                pass
        if os.path.exists(self.jsonl_path):
            cards = []
            with open(self.jsonl_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        cards.append(EvidenceCard.from_dict(json.loads(line)))
            return cards
        return []
