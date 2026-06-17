"""Generate research reports from evidence cards, articles, and article images."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config.news_config import load_research_config
from src.domain.news_taxonomy import normalize_topic
from src.research.article_image_store import ArticleImageStore
from src.research.article_summary import (
    build_article_display_summary,
    collect_article_industries,
    format_industry_label,
    kept_image_insights,
    parse_image_insights,
    resolve_image_insight,
)
from src.sources.wechat_article_content import normalize_wechat_article_url, split_digest_and_cover
from src.research.evidence_builder import EvidenceBuilder
from src.research.html_exporter import export_html_from_markdown
from src.research.report_planner import ReportPlan, ReportPlanner
from src.research.research_models import ArticleImage, ResearchReport
from src.research.research_note_builder import (
    build_daily_conclusions,
    build_watchlist,
    index_articles_by_url,
    resolve_event_display_text,
    sanitize_event_record,
)
from src.research.review_workflow import ReviewWorkflow
from src.research.weekly_synthesizer import (
    build_event_appendix,
    build_one_line_conclusion,
    build_quality_flags,
    build_weekly_themes,
    group_themes_by_topic,
)
from src.utils.news_storage import ensure_dir
from src.utils.news_text import coerce_list, coerce_str

logger = logging.getLogger(__name__)
UTC = timezone.utc
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates" / "reports"


class ReportGenerator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_research_config(config_path)
        research = self.config.get("research", {})
        self.output_dir = research.get("output_dir", "data/news/research")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        self.images_dir = os.path.join(self.output_dir, "article_images")
        self.reviewed_path = research.get("reviewed_all_path", "data/news/reviewed_all.csv")
        self.events_path = research.get("events_classified_path", "data/news/events_classified.csv")
        self.planner = ReportPlanner(self.config)
        self.evidence_builder = EvidenceBuilder(
            output_dir=os.path.join(self.output_dir, "evidence_cards"),
        )
        self.image_store = ArticleImageStore(output_dir=self.images_dir)
        self.review_workflow = ReviewWorkflow(
            state_path=os.path.join(self.output_dir, "review_state", "review_state.json"),
        )
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(
        self,
        report_type: str = "daily_brief",
        date: Optional[str] = None,
        days: int = 1,
        industry: Optional[str] = None,
        event_id: Optional[str] = None,
        include_images: bool = False,
    ) -> ResearchReport:
        plan = self.planner.plan(report_type, date=date, days=days, industry=industry)
        events_df = self._load_events()
        review_states = self.review_workflow.load_states()
        filtered = self.planner.filter_events(events_df, plan, review_states)

        if report_type == "single_event_flash" and event_id:
            filtered = events_df[events_df["event_id"] == event_id] if not events_df.empty else events_df

        events = [
            sanitize_event_record(event)
            for event in (filtered.to_dict(orient="records") if not filtered.empty else [])
        ]
        evidence_cards = [c.to_dict() for c in self.evidence_builder.load_cards()]
        evidence_cards = self.review_workflow.merge_into_cards(evidence_cards)
        event_ids = {str(e.get("event_id", "")) for e in events}
        cards_for_report = [c for c in evidence_cards if str(c.get("event_id", "")) in event_ids]

        articles = self._load_articles()
        articles_filtered = self.planner.filter_articles(articles, plan)

        report_article_ids = {
            str(a.get("article_id", ""))
            for a in articles_filtered
            if a.get("article_id")
        }
        article_images: list[ArticleImage] = []
        if include_images and self.config.get("article_images", {}).get("enabled", True):
            article_images = self.image_store.images_for_article_ids(report_article_ids)
            if not article_images and articles_filtered:
                article_images = self.image_store.build_for_articles(
                    articles_filtered, date_label=plan.date_label
                )
                article_images = [
                    img for img in article_images
                    if img.local_path and img.article_id in report_article_ids
                ]

        context = self._build_context(
            plan=plan,
            events=events,
            evidence_cards=cards_for_report,
            articles=articles_filtered,
            article_images=article_images,
            review_states=review_states,
            all_events_df=events_df,
        )

        template_name = f"{report_type}.md.j2"
        template = self.env.get_template(template_name)
        markdown = template.render(**context)

        report_dir = os.path.join(self.reports_dir, plan.date_label)
        ensure_dir(report_dir)
        md_path = os.path.join(report_dir, f"{report_type}.md")
        html_path = os.path.join(report_dir, f"{report_type}.html")

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)

        output_paths = [md_path]
        if "html" in self.config.get("research", {}).get("report_formats", ["markdown", "html"]):
            export_html_from_markdown(md_path, html_path, title=plan.title)
            output_paths.append(html_path)

        report_id = hashlib.sha256(f"{report_type}:{plan.date_label}".encode()).hexdigest()[:12]
        image_paths = [img.local_path for img in article_images if img.local_path]
        report_topics = [
            coerce_str(theme.get("topic", ""))
            for theme in context.get("weekly_theme_cards", [])
            if coerce_str(theme.get("topic", ""))
        ]
        report = ResearchReport(
            report_id=report_id,
            report_type=report_type,
            title=plan.title,
            period_start=plan.period_start.isoformat(),
            period_end=plan.period_end.isoformat(),
            industries=report_topics,
            etfs=[],
            generated_at=datetime.now(UTC).isoformat(),
            input_event_ids=[str(e.get("event_id", "")) for e in events],
            input_article_ids=[str(a.get("article_id", "")) for a in articles_filtered],
            image_paths=image_paths,
            output_paths=output_paths,
            summary=context.get("conclusions", [{}])[0].get("text", "") if context.get("conclusions") else "",
            status="generated",
        )

        meta_path = os.path.join(report_dir, f"{report_type}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)

        return report

    def _load_events(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in (self.reviewed_path, self.events_path):
            if not os.path.exists(path):
                continue
            try:
                frames.append(pd.read_csv(path))
            except Exception as exc:
                logger.warning(f"Failed to load {path}: {exc}")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if "event_id" in df.columns:
            df = df.drop_duplicates(subset=["event_id"], keep="last")
        return df

    def _load_articles(self) -> list[dict[str, Any]]:
        path = os.path.join(self.output_dir, "articles", "articles.parquet")
        if os.path.exists(path):
            try:
                return pd.read_parquet(path).to_dict(orient="records")
            except Exception:
                pass
        jsonl = os.path.join(self.output_dir, "articles", "articles.jsonl")
        articles = []
        if os.path.exists(jsonl):
            with open(jsonl, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        articles.append(json.loads(line))
        return articles

    @staticmethod
    def _materialize_report_image_path(
        source_path: str,
        *,
        report_dir: str,
        article_id: str,
        filename: str,
    ) -> str:
        """Copy archived images beside the report so local Markdown/HTML previews can load them."""
        src = Path(source_path)
        if not src.exists():
            return os.path.relpath(source_path, report_dir)

        dest_dir = ensure_dir(os.path.join(report_dir, "images", article_id))
        dest = Path(dest_dir) / filename
        if not dest.exists():
            shutil.copy2(src, dest)
        return os.path.join("images", article_id, filename)

    @staticmethod
    def _build_conclusions_with_images(
        events: list[dict[str, Any]],
        *,
        articles_by_url: dict[str, dict[str, Any]],
        groups_dict: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        conclusions = build_daily_conclusions(events, articles_by_url=articles_by_url)
        events_by_id = {
            coerce_str(event.get("event_id", "")): sanitize_event_record(event)
            for event in events
            if coerce_str(event.get("event_id", ""))
        }
        enriched: list[dict[str, Any]] = []
        for conclusion in conclusions:
            event = events_by_id.get(coerce_str(conclusion.get("event_id", "")))
            article = (
                articles_by_url.get(normalize_wechat_article_url(coerce_str(event.get("source_url", ""))))
                if event else None
            )
            article_id = coerce_str(article.get("article_id", "")) if article else ""
            item = dict(conclusion)
            item["title"] = (
                coerce_str(event.get("title", ""))
                if event else coerce_str(article.get("title", "")) if article else ""
            )
            item["images"] = list(groups_dict.get(article_id, {}).get("images", []))
            enriched.append(item)
        return enriched

    def _build_context(
        self,
        plan: ReportPlan,
        events: list[dict[str, Any]],
        evidence_cards: list[dict[str, Any]],
        articles: list[dict[str, Any]],
        article_images: list[ArticleImage],
        review_states: dict[str, dict],
        all_events_df: pd.DataFrame,
    ) -> dict[str, Any]:
        articles_by_url = index_articles_by_url(articles)
        watchlist = build_watchlist(evidence_cards, review_states)

        report_dir = os.path.join(self.reports_dir, plan.date_label)
        articles_by_id = {str(a.get("article_id", "")): a for a in articles if a.get("article_id")}
        events_by_id = {
            coerce_str(e.get("event_id", "")): sanitize_event_record(e)
            for e in all_events_df.to_dict(orient="records")
            if coerce_str(e.get("event_id", ""))
        }

        index_paths_by_article: dict[str, list[str]] = {}
        for image in article_images:
            if image.local_path:
                index_paths_by_article.setdefault(image.article_id, []).append(image.local_path)

        image_blocks = []
        groups_dict: dict[str, dict[str, Any]] = {}
        for image in article_images:
            if not image.local_path:
                continue
            art = articles_by_id.get(image.article_id, {})
            insight = resolve_image_insight(
                art,
                local_path=image.local_path,
                image_index=image.index,
            )
            rel = self._materialize_report_image_path(
                image.local_path,
                report_dir=report_dir,
                article_id=image.article_id,
                filename=Path(image.local_path).name,
            )
            block = {
                "path": rel,
                "article_id": image.article_id,
                "event_ids": image.event_ids,
                "fetch_status": image.fetch_status,
                "original_url": image.original_url,
                "explanation": coerce_str(insight.get("explanation", "")),
                "keep": bool(insight.get("keep", True)) if insight else True,
            }
            image_blocks.append(block)
            if image.article_id not in groups_dict:
                groups_dict[image.article_id] = {
                    "article_id": image.article_id,
                    "title": art.get("title", ""),
                    "source_name": art.get("source_name", ""),
                    "source_url": art.get("source_url", ""),
                    "images": [],
                }
            groups_dict[image.article_id]["images"].append(block)
        article_image_groups = list(groups_dict.values())

        conclusions = self._build_conclusions_with_images(
            events,
            articles_by_url=articles_by_url,
            groups_dict=groups_dict,
        )

        articles_by_source: dict[str, list[dict]] = {}
        for article in articles:
            source = str(article.get("source_name", "未知来源"))
            article_id = str(article.get("article_id", ""))
            raw_paths = article.get("image_paths", [])
            if raw_paths is None:
                image_paths: list[str] = []
            elif isinstance(raw_paths, list):
                image_paths = [str(p) for p in raw_paths if p]
            else:
                try:
                    image_paths = [str(p) for p in list(raw_paths) if p]
                except TypeError:
                    image_paths = [str(raw_paths)] if raw_paths else []
            if not image_paths:
                image_paths = index_paths_by_article.get(article_id, [])
            digest, cover_from_digest = split_digest_and_cover(coerce_str(article.get("digest")))
            cover_image_url = coerce_str(article.get("cover_image_url")) or cover_from_digest
            related_events = [
                events_by_id[eid]
                for eid in coerce_list(article.get("related_event_ids"))
                if eid in events_by_id
            ]
            industries = collect_article_industries(article, related_events)
            image_insights = parse_image_insights(article.get("agent_image_insights"))
            kept_insights = kept_image_insights(article.get("agent_image_insights"))
            articles_by_source.setdefault(source, []).append({
                "title": article.get("title", ""),
                "published_at": article.get("published_at", ""),
                "summary": build_article_display_summary(article, related_events),
                "digest": digest,
                "cover_image_url": cover_image_url,
                "industries": industries,
                "industry_labels": [format_industry_label(i) for i in industries],
                "in_report": True,
                "url": article.get("source_url", ""),
                "image_paths": image_paths,
                "images_fetch_status": article.get("images_fetch_status", "pending"),
                "image_insights": image_insights,
                "kept_image_insights": kept_insights,
            })

        event_table = []
        for event in events:
            eid = coerce_str(event.get("event_id", ""))
            review = review_states.get(eid, {})
            needs_review = review.get("status", "new") not in {"verified", "report_ready"}
            article = articles_by_url.get(normalize_wechat_article_url(coerce_str(event.get("source_url", ""))))
            event_table.append({
                "topic": normalize_topic(event.get("topic")) or normalize_topic(event.get("industry")) or "未分类",
                "event_name": coerce_str(event.get("event_name")) or coerce_str(event.get("title", "")),
                "title": coerce_str(event.get("title", "")),
                "source": coerce_str(event.get("source_name", "")),
                "reasoning": coerce_str(event.get("agent_reasoning", "")),
                "summary": resolve_event_display_text(event, article=article, max_chars=0),
                "needs_review": needs_review,
                "event_id": eid,
                "source_url": coerce_str(event.get("source_url", "")),
                "published_at": coerce_str(event.get("event_date", "")),
            })

        weekly_theme_cards = []
        weekly_quality_flags = []
        weekly_event_appendix = {}
        weekly_one_line_conclusion = ""
        weekly_themes_by_topic = {}
        if plan.report_type in {"daily_brief", "weekly_review"}:
            theme_limit = 5 if plan.report_type == "daily_brief" else 8
            weekly_theme_cards = build_weekly_themes(
                events,
                articles_by_url=articles_by_url,
                max_themes=theme_limit,
            )
            weekly_quality_flags = build_quality_flags(events)
            weekly_event_appendix = build_event_appendix(events)
            weekly_one_line_conclusion = build_one_line_conclusion(
                weekly_theme_cards,
                weekly_event_appendix,
            )
            weekly_themes_by_topic = group_themes_by_topic(weekly_theme_cards)

        weekly_themes = self._extract_weekly_themes(all_events_df, plan) if plan.report_type == "weekly_review" else []
        industry_sections = self._industry_sections(events) if plan.report_type == "industry_report" else {}

        return {
            "title": plan.title,
            "date_label": plan.date_label,
            "period_start": plan.period_start.strftime("%Y-%m-%d"),
            "period_end": plan.period_end.strftime("%Y-%m-%d"),
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "conclusions": conclusions,
            "event_table": event_table,
            "industry_summary": {},
            "articles_by_source": articles_by_source,
            "article_images": image_blocks,
            "article_image_groups": article_image_groups,
            "evidence_cards": evidence_cards,
            "watchlist": watchlist,
            "weekly_themes": weekly_themes,
            "weekly_theme_cards": weekly_theme_cards,
            "weekly_themes_by_etf": weekly_themes_by_topic,
            "weekly_themes_by_topic": weekly_themes_by_topic,
            "weekly_quality_flags": weekly_quality_flags,
            "weekly_event_appendix": weekly_event_appendix,
            "weekly_one_line_conclusion": weekly_one_line_conclusion,
            "industry_sections": industry_sections,
            "industry": plan.industries[0] if plan.industries else "",
            "events": events,
            "articles": articles,
            "rejected_appendix": self._rejected_appendix(review_states, evidence_cards),
        }

    def _extract_weekly_themes(self, df: pd.DataFrame, plan: ReportPlan) -> list[str]:
        if df.empty:
            return []
        sub = df.copy()
        if "event_date" in sub.columns:
            sub["event_date"] = pd.to_datetime(sub["event_date"], format="mixed", utc=True, errors="coerce")
            sub = sub[
                (sub["event_date"] >= plan.period_start)
                & (sub["event_date"] <= plan.period_end)
            ]
        if "topic" in sub.columns:
            topics = sub["topic"].map(lambda value: normalize_topic(coerce_str(value)))
        elif "industry" in sub.columns:
            topics = sub["industry"].map(lambda value: normalize_topic(coerce_str(value)))
        else:
            return []
        topics = topics[topics != ""]
        if topics.empty:
            return []
        counts = topics.value_counts()
        return [f"{idx}（{cnt}条）" for idx, cnt in counts.items()]

    def _industry_sections(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, list[dict]]:
        sections: dict[str, list[dict]] = {}
        for event in events:
            topic = normalize_topic(event.get("topic")) or normalize_topic(event.get("industry")) or "未分类"
            sections.setdefault(topic, []).append(event)
        return sections

    def _rejected_appendix(
        self,
        review_states: dict[str, dict],
        evidence_cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.config.get("research", {}).get("include_rejected_in_appendix", False):
            return []
        rejected_ids = {eid for eid, st in review_states.items() if st.get("status") == "rejected"}
        return [c for c in evidence_cards if str(c.get("event_id", "")) in rejected_ids]
