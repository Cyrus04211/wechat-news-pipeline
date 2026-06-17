#!/usr/bin/env python3
"""WeChat research message collection and LLM processing pipeline."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from src.config.news_config import load_llm_config, load_runtime_config
from src.pipelines.news_collection_profiles import COLLECTION_MODES, run_collection_mode


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_dir = load_runtime_config().get("log_dir", "data/news/logs")
    os.makedirs(log_dir, exist_ok=True)

    from datetime import datetime
    log_file = os.path.join(log_dir, f"news_collect_{datetime.now().strftime('%Y-%m-%d')}.log")

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def show_recent(parquet_path: str, hours: int) -> None:
    if not os.path.exists(parquet_path):
        print(f"No output file found at {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    time_col = "event_date"
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], format='mixed', utc=True)
        df = df[df[time_col] >= cutoff]
    if "is_noise" in df.columns:
        df = df[df["is_noise"] == False]
    df = df.sort_values(time_col, ascending=False) if time_col in df.columns else df

    if df.empty:
        print(f"No classified events in the last {hours}h")
        return

    print(f"\n=== Topic-Classified Events (last {hours}h) ===")
    print(f"{'Date':<22} {'Topic':<24} {'Source':<16} Event")
    print("-" * 120)
    for _, row in df.iterrows():
        date_str = str(row.get("event_date", ""))[:19]
        print(
            f"{date_str:<22} {str(row.get('topic', '') or '未分类'):<24} "
            f"{str(row.get('source_name', '')):<16} {row.get('event_name') or row.get('title', '')}"
        )
    print(f"\nTotal: {len(df)} events\n")


def main():
    parser = argparse.ArgumentParser(
        description="公众号行业调研消息采集与 LLM 摘要/主题分类",
    )
    parser.add_argument("--show-recent", action="store_true",
                        help="Show recent classified events from existing output")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours for --show-recent filter")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--agent-provider", type=str, default=None,
                        choices=["auto", "closeai", "openai-compatible"],
                        help="LLM provider")
    parser.add_argument("--agent-model", type=str, default=None,
                        help="Model for inline API mode")
    parser.add_argument("--agent-api-key", type=str, default=None,
                        help="Inline API key")
    parser.add_argument("--agent-base-url", type=str, default=None,
                        help="Override inline API base URL")
    parser.add_argument("--llm-batch-size", type=int, default=12,
                        help="Items per LLM API call")
    parser.add_argument(
        "--collection-mode",
        type=str,
        default="daily_brief",
        choices=list(COLLECTION_MODES),
        help="daily_brief=24h；weekly_review=168h",
    )
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    config = load_runtime_config()
    llm_config = load_llm_config()
    parquet_path = config.get("output_parquet", "data/news/events_classified.parquet")

    if args.show_recent:
        show_recent(parquet_path, hours=args.hours)
        return 0

    results = run_collection_mode(
        args.collection_mode,
        debug=args.debug,
        agent_provider=args.agent_provider,
        agent_api_key=args.agent_api_key,
        agent_model=args.agent_model,
        agent_base_url=args.agent_base_url,
        llm_batch_size=args.llm_batch_size,
    )
    if not results:
        raise SystemExit(f"No configured profiles available for mode: {args.collection_mode}")
    for profile_name, count in results.items():
        logging.getLogger(__name__).info("Profile %s produced %s retained events", profile_name, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
