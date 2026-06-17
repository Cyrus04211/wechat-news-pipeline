#!/usr/bin/env python3
"""CLI: manual review workflow for research evidence cards."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.news_config import load_research_config
from src.research.review_workflow import ReviewWorkflow


def main() -> int:
    parser = argparse.ArgumentParser(description="调研证据人工复核")
    parser.add_argument("--list", action="store_true", help="列出待复核事件")
    parser.add_argument("--status", default=None, help="筛选状态 new/watch/verified/rejected/report_ready")
    parser.add_argument("--set-status", nargs=2, metavar=("EVENT_ID", "STATUS"), help="设置复核状态")
    parser.add_argument("--note", default="", help="复核备注")
    parser.add_argument("--export-watchlist", action="store_true", help="导出待跟踪清单 CSV")
    args = parser.parse_args()

    config = load_research_config()
    research = config.get("research", {})
    output_dir = research.get("output_dir", "data/news/research")
    state_path = os.path.join(output_dir, "review_state", "review_state.json")
    evidence_path = os.path.join(output_dir, "evidence_cards", "evidence_cards.csv")
    workflow = ReviewWorkflow(state_path=state_path)

    if args.set_status:
        event_id, status = args.set_status
        entry = workflow.set_status(event_id, status, note=args.note)
        print(f"已更新 {event_id} -> {status}")
        if entry.get("note"):
            print(f"  备注: {entry['note']}")
        return 0

    if args.export_watchlist:
        path = workflow.export_watchlist(
            output_path=os.path.join(output_dir, "review_state", "watchlist.csv"),
        )
        print(f"已导出待跟踪清单: {path}")
        return 0

    if args.list:
        items = workflow.list_events(
            status=args.status,
            evidence_cards_path=evidence_path,
        )
        if not items:
            print("无匹配事件")
            return 0
        print(f"{'event_id':<18} {'status':<12} {'industry':<16} title")
        print("-" * 100)
        for item in items:
            print(
                f"{item['event_id']:<18} {item['status']:<12} "
                f"{str(item.get('industry', '')):<16} {str(item.get('title', ''))}"
            )
        print(f"\n共 {len(items)} 条")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
