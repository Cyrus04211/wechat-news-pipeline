#!/usr/bin/env python3
"""CLI: generate static research dashboard HTML."""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.news_config import load_research_config
from src.research.article_image_store import ArticleImageStore
from src.research.evidence_builder import EvidenceBuilder
from src.research.review_workflow import ReviewWorkflow
from src.utils.news_storage import ensure_dir

UTC = timezone.utc
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates" / "reports"


def build_dashboard_context(
    output_dir: str,
    days: int = 7,
    reviewed_path: str = "data/news/reviewed_all.csv",
    events_path: str = "data/news/events_classified.csv",
) -> dict:
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=days)

    events_df = pd.DataFrame()
    for path in (reviewed_path, events_path):
        if os.path.exists(path):
            try:
                events_df = pd.read_csv(path)
                break
            except Exception:
                pass

    if not events_df.empty and "event_date" in events_df.columns:
        events_df["event_date"] = pd.to_datetime(events_df["event_date"], format="mixed", utc=True, errors="coerce")
        recent_events = events_df[events_df["event_date"] >= cutoff]
    else:
        recent_events = events_df

    if not recent_events.empty and "is_noise" in recent_events.columns:
        recent_events = recent_events[recent_events["is_noise"] == False]
    classified_events = []
    if not recent_events.empty:
        for _, row in recent_events.iterrows():
            classified_events.append({
                "event_id": row.get("event_id", ""),
                "title": row.get("event_name", "") or row.get("title", ""),
                "topic": row.get("topic", "") or "未分类",
                "source_name": row.get("source_name", ""),
                "source_url": row.get("source_url", ""),
                "reasoning": row.get("agent_reasoning", ""),
            })

    review = ReviewWorkflow(state_path=os.path.join(output_dir, "review_state", "review_state.json"))
    pending = review.list_events(status="new")
    pending += [p for p in review.list_events(status="watch") if p not in pending]

    source_counts = []
    if not recent_events.empty and "source_name" in recent_events.columns:
        counts = recent_events["source_name"].value_counts()
        source_counts = [{"source": k, "count": int(v)} for k, v in counts.items()]

    topic_summary: dict[str, int] = {}
    for event in classified_events:
        topic = event.get("topic", "") or "未分类"
        topic_summary[topic] = topic_summary.get(topic, 0) + 1

    image_store = ArticleImageStore(output_dir=os.path.join(output_dir, "article_images"))
    image_items = []
    dashboards_dir = os.path.join(output_dir, "dashboards")
    for image in image_store.load_index():
        if not image.local_path:
            continue
        rel = os.path.relpath(image.local_path, dashboards_dir)
        image_items.append({
            "path": rel,
            "article_id": image.article_id,
            "fetch_status": image.fetch_status,
        })

    reports_dir = os.path.join(output_dir, "reports")
    report_links = []
    if os.path.isdir(reports_dir):
        for day_dir in sorted(Path(reports_dir).iterdir(), reverse=True):
            if not day_dir.is_dir():
                continue
            for report_file in day_dir.glob("*.html"):
                report_links.append({
                    "date": day_dir.name,
                    "name": report_file.stem,
                    "path": os.path.relpath(str(report_file), dashboards_dir),
                })

    metrics = {
        "total_events": len(recent_events),
        "classified_count": len(classified_events),
        "pending_review": len(pending),
        "source_count": len(source_counts),
        "image_count": len(image_items),
        "days": days,
    }

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
        "period_label": f"最近 {days} 天",
        "metrics": metrics,
        "topic_summary": topic_summary,
        "classified_events": classified_events,
        "pending_review": pending,
        "source_counts": source_counts,
        "article_images": image_items,
        "report_links": report_links,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成调研 Dashboard 静态页面")
    parser.add_argument("--days", type=int, default=7, help="统计回溯天数")
    args = parser.parse_args()

    config = load_research_config()
    research = config.get("research", {})
    output_dir = research.get("output_dir", "data/news/research")
    events_path = research.get("events_classified_path", "data/news/events_classified.csv")
    reviewed_path = research.get("reviewed_all_path", "data/news/reviewed_all.csv")

    evidence_path = os.path.join(output_dir, "evidence_cards", "evidence_cards.csv")
    if not os.path.exists(evidence_path):
        EvidenceBuilder(
            output_dir=os.path.join(output_dir, "evidence_cards"),
        ).build(reviewed_path=reviewed_path, events_path=events_path)

    context = build_dashboard_context(
        output_dir=output_dir,
        days=args.days,
        reviewed_path=reviewed_path,
        events_path=events_path,
    )

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("dashboard.html.j2")
    html = template.render(**context)

    dashboards_dir = os.path.join(output_dir, "dashboards")
    ensure_dir(dashboards_dir)
    out_path = os.path.join(dashboards_dir, "latest.html")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(html)

    meta_path = os.path.join(dashboards_dir, "latest_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"generated_at": context["generated_at"], "metrics": context["metrics"]}, handle, ensure_ascii=False, indent=2)

    print(f"Dashboard 已生成: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
