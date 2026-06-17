"""Plan report scope, filters, and section inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, Optional

import pandas as pd

UTC = timezone.utc


@dataclass
class ReportPlan:
    report_type: str
    title: str
    period_start: datetime
    period_end: datetime
    industries: list[str] = field(default_factory=list)
    date_label: str = ""
    include_rejected: bool = False


class ReportPlanner:
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        research = self.config.get("research", {})
        self.include_rejected = bool(research.get("include_rejected_in_appendix", False))
        self.timezone = research.get("timezone", "Asia/Shanghai")

    def plan(
        self,
        report_type: str,
        date: Optional[str] = None,
        days: int = 1,
        industry: Optional[str] = None,
    ) -> ReportPlan:
        if date:
            end_day = datetime.fromisoformat(date).date()
        else:
            end_day = datetime.now(UTC).date()

        start_day = end_day - timedelta(days=max(days - 1, 0))
        period_start = datetime.combine(start_day, time.min, tzinfo=UTC)
        period_end = datetime.combine(end_day, time.max.replace(microsecond=0), tzinfo=UTC)
        date_label = end_day.strftime("%Y-%m-%d")

        industries: list[str] = []

        titles = {
            "daily_brief": f"公众号行业调研日报：{date_label}",
            "weekly_review": f"公众号行业调研周报：{start_day.strftime('%Y-%m-%d')} ~ {date_label}",
            "industry_report": f"主题专题：{industry or 'all'} ({date_label})",
            "single_event_flash": f"单事件快评 ({date_label})",
            "source_digest": f"公众号文章整理：{start_day.strftime('%Y-%m-%d')} ~ {date_label}",
        }

        return ReportPlan(
            report_type=report_type,
            title=titles.get(report_type, f"研究报告 {date_label}"),
            period_start=period_start,
            period_end=period_end,
            industries=industries,
            date_label=date_label,
            include_rejected=self.include_rejected,
        )

    def filter_events(
        self,
        events_df: pd.DataFrame,
        plan: ReportPlan,
        review_status: Optional[dict[str, dict]] = None,
    ) -> pd.DataFrame:
        if events_df.empty:
            return events_df

        df = events_df.copy()
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], format="mixed", utc=True, errors="coerce")
            df = df[
                (df["event_date"] >= plan.period_start)
                & (df["event_date"] <= plan.period_end)
            ]

        review_status = review_status or {}
        if not plan.include_rejected:
            rejected_ids = {
                eid for eid, st in review_status.items()
                if st.get("status") == "rejected"
            }
            if rejected_ids and "event_id" in df.columns:
                df = df[~df["event_id"].isin(rejected_ids)]

        if "event_date" in df.columns:
            df = df.sort_values("event_date", ascending=False)
        return df

    def filter_articles(
        self,
        articles: list[dict[str, Any]],
        plan: ReportPlan,
    ) -> list[dict[str, Any]]:
        result = []
        for article in articles:
            published = str(article.get("published_at", "") or "")
            if published:
                try:
                    pub_dt = pd.to_datetime(published, format="mixed", utc=True)
                    if pub_dt < plan.period_start or pub_dt > plan.period_end:
                        continue
                except Exception:
                    pass
            result.append(article)
        return result

    def etf_list(self) -> list[str]:
        return []
