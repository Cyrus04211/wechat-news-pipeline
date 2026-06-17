"""Lightweight dataclass models for research reporting layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from src.utils.news_text import coerce_list, coerce_str


@dataclass
class ResearchArticle:
    article_id: str
    source_id: str = ""
    source_name: str = ""
    source_tier: int = 0
    title: str = ""
    author: str = ""
    published_at: str = ""
    collected_at: str = ""
    source_url: str = ""
    industry_candidates: list[str] = field(default_factory=list)
    digest: str = ""
    agent_summary: str = ""
    agent_image_insights: str = ""
    cover_image_url: str = ""
    content_text: str = ""
    content_html: str = ""
    content_markdown: str = ""
    content_hash: str = ""
    fetch_status: str = "success"  # success / partial / failed / manual_import
    images_fetch_status: str = "pending"  # pending / ok / no_images / failed
    image_paths: list[str] = field(default_factory=list)
    raw_path: str = ""
    related_event_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchArticle:
        return cls(
            article_id=coerce_str(data.get("article_id", "")),
            source_id=coerce_str(data.get("source_id", "")),
            source_name=coerce_str(data.get("source_name", "")),
            source_tier=int(data.get("source_tier", 0) or 0),
            title=coerce_str(data.get("title", "")),
            author=coerce_str(data.get("author", "")),
            published_at=coerce_str(data.get("published_at", "")),
            collected_at=coerce_str(data.get("collected_at", "")),
            source_url=coerce_str(data.get("source_url", "")),
            industry_candidates=coerce_list(data.get("industry_candidates", [])),
            digest=coerce_str(data.get("digest", "")),
            agent_summary=coerce_str(data.get("agent_summary", "")),
            agent_image_insights=coerce_str(data.get("agent_image_insights", "")),
            cover_image_url=coerce_str(data.get("cover_image_url", "")),
            content_text=coerce_str(data.get("content_text", "")),
            content_html=coerce_str(data.get("content_html", "")),
            content_markdown=coerce_str(data.get("content_markdown", "")),
            content_hash=coerce_str(data.get("content_hash", "")),
            fetch_status=coerce_str(data.get("fetch_status", "success")) or "success",
            images_fetch_status=coerce_str(data.get("images_fetch_status", "pending")) or "pending",
            image_paths=coerce_list(data.get("image_paths", [])),
            raw_path=coerce_str(data.get("raw_path", "")),
            related_event_ids=coerce_list(data.get("related_event_ids", [])),
            tags=coerce_list(data.get("tags", [])),
            notes=coerce_str(data.get("notes", "")),
        )


@dataclass
class ArticleImage:
    image_id: str
    article_id: str = ""
    event_ids: list[str] = field(default_factory=list)
    source_url: str = ""
    original_url: str = ""
    local_path: str = ""
    fetch_status: str = "pending"
    date_label: str = ""
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArticleImage:
        return cls(
            image_id=str(data.get("image_id", "")),
            article_id=str(data.get("article_id", "") or ""),
            event_ids=coerce_list(data.get("event_ids", [])),
            source_url=str(data.get("source_url", "") or ""),
            original_url=str(data.get("original_url", "") or ""),
            local_path=str(data.get("local_path", "") or ""),
            fetch_status=str(data.get("fetch_status", "pending")),
            date_label=str(data.get("date_label", "") or ""),
            index=int(data.get("index", 0) or 0),
        )


@dataclass
class EvidenceCard:
    evidence_id: str
    event_id: str
    article_id: str = ""
    title: str = ""
    source_name: str = ""
    source_url: str = ""
    published_at: str = ""
    industry: str = ""
    etf: str = ""
    key_claims: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    evidence_quality_score: float = 0.0
    verification_questions: list[str] = field(default_factory=list)
    follow_up_actions: list[str] = field(default_factory=list)
    reviewer_status: str = "new"
    reviewer_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceCard:
        return cls(
            evidence_id=coerce_str(data.get("evidence_id", "")),
            event_id=coerce_str(data.get("event_id", "")),
            article_id=coerce_str(data.get("article_id", "")),
            title=coerce_str(data.get("title", "")),
            source_name=coerce_str(data.get("source_name", "")),
            source_url=coerce_str(data.get("source_url", "")),
            published_at=coerce_str(data.get("published_at", "")),
            industry=coerce_str(data.get("industry", "")),
            etf=coerce_str(data.get("etf", "")),
            key_claims=coerce_list(data.get("key_claims", [])),
            supporting_evidence=coerce_list(data.get("supporting_evidence", [])),
            evidence_quality_score=float(data.get("evidence_quality_score", 0) or 0),
            verification_questions=coerce_list(data.get("verification_questions", [])),
            follow_up_actions=coerce_list(data.get("follow_up_actions", [])),
            reviewer_status=coerce_str(data.get("reviewer_status", "new")) or "new",
            reviewer_notes=coerce_str(data.get("reviewer_notes", "")),
        )


@dataclass
class ResearchReport:
    report_id: str
    report_type: str
    title: str = ""
    period_start: str = ""
    period_end: str = ""
    industries: list[str] = field(default_factory=list)
    etfs: list[str] = field(default_factory=list)
    generated_at: str = ""
    input_event_ids: list[str] = field(default_factory=list)
    input_article_ids: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    summary: str = ""
    status: str = "generated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchReport:
        return cls(
            report_id=str(data.get("report_id", "")),
            report_type=str(data.get("report_type", "")),
            title=str(data.get("title", "")),
            period_start=str(data.get("period_start", "") or ""),
            period_end=str(data.get("period_end", "") or ""),
            industries=coerce_list(data.get("industries", [])),
            etfs=coerce_list(data.get("etfs", [])),
            generated_at=str(data.get("generated_at", "") or ""),
            input_event_ids=coerce_list(data.get("input_event_ids", [])),
            input_article_ids=coerce_list(data.get("input_article_ids", [])),
            image_paths=coerce_list(data.get("image_paths", data.get("chart_paths", []))),
            output_paths=coerce_list(data.get("output_paths", [])),
            summary=str(data.get("summary", "") or ""),
            status=str(data.get("status", "generated")),
        )
