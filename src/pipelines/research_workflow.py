#!/usr/bin/env python3
"""CLI: end-to-end research workflow (collect -> archive -> evidence -> images -> report)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.news_config import load_research_config, load_llm_config, load_runtime_config
from src.pipelines.news_collection_profiles import COLLECTION_MODES, resolve_report_plan, run_collection_mode
from src.research.article_repository import ArticleRepository, article_fetch_kwargs_from_runtime
from src.research.evidence_builder import EvidenceBuilder
from src.research.image_insight_backfill import ImageInsightBackfill
from src.research.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


def run_collect(collection_mode: str) -> None:
    llm_config = load_llm_config()
    results = run_collection_mode(
        collection_mode,
        agent_provider=llm_config.get("provider", "auto"),
        agent_model=llm_config.get("model", "deepseek-v4-pro"),
        agent_base_url=llm_config.get("api_base"),
    )
    if not results:
        logger.warning("采集未返回结果，将继续使用已有数据生成研报")
    else:
        for profile, count in results.items():
            print(f"采集配置 {profile}: {count} 条主题分类事件（全部写入汇总）")


def main() -> int:
    parser = argparse.ArgumentParser(description="调研工作流：采集主题分类 + 归档 + 证据卡片 + 研报")
    parser.add_argument(
        "--mode",
        default="report-only",
        choices=["collect-classify-report", "archive-report", "report-only"],
        help="collect-classify-report 先采集分类；archive-report 仅归档与生成；report-only 仅生成研报",
    )
    parser.add_argument(
        "--collection-mode",
        default="daily_brief",
        choices=list(COLLECTION_MODES),
        help="daily_brief=24h 日报；weekly_review=168h 周报",
    )
    parser.add_argument(
        "--report-type",
        default=None,
        choices=["daily_brief", "weekly_review", "industry_report", "single_event_flash", "source_digest"],
        help="研报模板；省略时按采集窗口天数自动选择（1天→daily_brief，多天→weekly_review）",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--days", type=int, default=None, help="研报覆盖天数；省略时与 collection-mode 的 lookback 对齐")
    parser.add_argument("--industry", default=None, help="主题专题报告的主题名（兼容旧参数名）")
    parser.add_argument(
        "--skip-image-backfill",
        action="store_true",
        help="跳过研报生成前的图片解读回填（默认会补齐缺失的 LLM 图片分析）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_research_config()
    research = config.get("research", {})
    output_dir = research.get("output_dir", "data/news/research")

    if args.mode == "collect-classify-report":
        print("步骤 1/5: 采集与 LLM 摘要及主题分类...")
        run_collect(args.collection_mode)

    if args.mode in {"collect-classify-report", "archive-report"}:
        print("步骤 2/5: 构建文章索引（含正文与原文图片）...")
        runtime = load_runtime_config()
        runtime["fetch_article_content"] = True
        repo = ArticleRepository(output_dir=os.path.join(output_dir, "articles"))
        repo.build_from_pipeline(
            reviewed_path=research.get("reviewed_all_path", "data/news/reviewed_all.csv"),
            events_path=research.get("events_classified_path", "data/news/events_classified.csv"),
            raw_dir=research.get("raw_dir", "data/news/raw"),
            **article_fetch_kwargs_from_runtime(runtime, config),
        )

        print("步骤 3/5: 构建证据卡片...")
        EvidenceBuilder(
            output_dir=os.path.join(output_dir, "evidence_cards"),
        ).build(
            reviewed_path=research.get("reviewed_all_path", "data/news/reviewed_all.csv"),
            events_path=research.get("events_classified_path", "data/news/events_classified.csv"),
        )

    report_type, days, lookback_hours = resolve_report_plan(
        collection_mode=args.collection_mode,
        report_type=args.report_type,
        days=args.days,
    )
    if args.report_type is None or args.days is None:
        print(
            f"研报窗口随采集配置：collection-mode={args.collection_mode} "
            f"lookback={lookback_hours}h → days={days} report-type={report_type}"
        )

    backfill_cfg = config.get("image_insight_backfill", {})
    if (
        not args.skip_image_backfill
        and backfill_cfg.get("enabled", True)
        and args.mode in {"collect-classify-report", "archive-report", "report-only"}
    ):
        print("步骤 3.5/5: 回填缺失的图片解读（分批 Vision）...")
        backfill = ImageInsightBackfill(
            articles_dir=os.path.join(output_dir, "articles"),
            images_dir=research.get("article_images_dir", os.path.join(output_dir, "article_images")),
            vision_batch_size=int(backfill_cfg.get("vision_batch_size", 4)),
            sync_event_tables=bool(backfill_cfg.get("sync_event_tables", True)),
            reviewed_path=research.get("reviewed_all_path", "data/news/reviewed_all.csv"),
            events_path=research.get("events_classified_path", "data/news/events_classified.csv"),
        )
        backfill_stats = backfill.run()
        pending = backfill_stats.get("pending", 0)
        print(
            "图片解读回填: "
            f"检查 {backfill_stats['checked']} 篇，更新 {backfill_stats['updated']} 篇，"
            f"跳过 {backfill_stats['skipped']} 篇，失败 {backfill_stats['failed']} 篇"
            + (f"，待 API 回填 {pending} 篇" if pending else "")
        )

    print("步骤 4/5: 生成研报（引用原文图片）...")
    report = ReportGenerator().generate(
        report_type=report_type,
        date=args.date,
        days=days,
        industry=args.industry,
    )

    print("\n步骤 5/5: 完成")
    print(f"研报: {report.title}")
    for path in report.output_paths:
        print(f"  {path}")
    if report.image_paths:
        print(f"  原文图片: {len(report.image_paths)} 张")
    evidence_csv = os.path.join(output_dir, "evidence_cards", "evidence_cards.csv")
    if os.path.exists(evidence_csv):
        print(f"  证据卡片: {evidence_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
