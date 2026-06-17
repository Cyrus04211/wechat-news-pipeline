#!/usr/bin/env python3
"""CLI: generate research reports without full collection."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.news_config import load_research_config, load_runtime_config
from src.research.article_repository import ArticleRepository, article_fetch_kwargs_from_runtime
from src.research.evidence_builder import EvidenceBuilder
from src.research.report_generator import ReportGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description="生成调研研报（日报/周报/主题专题/快评/公众号整理）")
    parser.add_argument(
        "--report-type",
        required=True,
        choices=["daily_brief", "weekly_review", "industry_report", "single_event_flash", "source_digest"],
    )
    parser.add_argument("--date", default=None, help="报告日期 YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=None, help="回溯天数")
    parser.add_argument("--industry", default=None, help="主题专题报告的主题名（兼容旧参数名）")
    parser.add_argument("--event-id", default=None, help="单事件快评的事件 ID")
    parser.add_argument("--skip-evidence", action="store_true", help="跳过证据卡片重建")
    parser.add_argument("--skip-articles", action="store_true", help="跳过文章索引重建")
    args = parser.parse_args()

    config = load_research_config()
    research = config.get("research", {})
    output_dir = research.get("output_dir", "data/news/research")
    events_path = research.get("events_classified_path", "data/news/events_classified.csv")
    reviewed_path = research.get("reviewed_all_path", "data/news/reviewed_all.csv")

    if not args.skip_articles:
        runtime = load_runtime_config()
        runtime["fetch_article_content"] = True
        repo = ArticleRepository(output_dir=os.path.join(output_dir, "articles"))
        articles = repo.build_from_pipeline(
            reviewed_path=reviewed_path,
            events_path=events_path,
            raw_dir=research.get("raw_dir", "data/news/raw"),
            **article_fetch_kwargs_from_runtime(runtime, config),
        )
        print(f"文章索引已更新: {len(articles)} 篇")

    if not args.skip_evidence:
        builder = EvidenceBuilder(output_dir=os.path.join(output_dir, "evidence_cards"))
        cards = builder.build(reviewed_path=reviewed_path, events_path=events_path)
        print(f"证据卡片已更新: {len(cards)} 条")

    days = args.days
    if days is None:
        if args.report_type == "weekly_review":
            days = int(research.get("default_weekly_days", 7))
        else:
            days = int(research.get("default_report_days", 1))

    generator = ReportGenerator()
    report = generator.generate(
        report_type=args.report_type,
        date=args.date,
        days=days,
        industry=args.industry,
        event_id=args.event_id,
    )
    print(f"\n研报已生成: {report.title}")
    for path in report.output_paths:
        print(f"  - {path}")
    if report.image_paths:
        print(f"  原文图片: {len(report.image_paths)} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
