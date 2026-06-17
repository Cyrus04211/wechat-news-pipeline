#!/usr/bin/env python3
"""CLI: manually import WeChat article content into research archive."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.news_config import load_research_config
from src.research.article_repository import ArticleRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="手动导入公众号文章到调研资料库")
    parser.add_argument("--source-name", required=True, help="来源名称，如「手动导入」")
    parser.add_argument("--industry", default="", help="行业代码 semiconductor/aerospace/minor_metals/lithium_battery")
    parser.add_argument("--file", required=True, help="文章文件路径 .md/.txt/.html")
    parser.add_argument("--url", default="", help="可选原文链接")
    args = parser.parse_args()

    config = load_research_config()
    output_dir = os.path.join(
        config.get("research", {}).get("output_dir", "data/news/research"),
        "articles",
    )
    repo = ArticleRepository(output_dir=output_dir)
    article = repo.import_manual(
        file_path=args.file,
        source_name=args.source_name,
        industry=args.industry,
        url=args.url,
    )
    print(f"已导入文章: {article.article_id}")
    print(f"  标题: {article.title}")
    print(f"  状态: {article.fetch_status}")
    print(f"  存档: {repo.parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
