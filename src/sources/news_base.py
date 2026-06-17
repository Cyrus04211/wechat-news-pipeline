import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class RawNewsItem:
    source_name: str
    source_tier: int
    title: str
    event_date: datetime
    source_url: Optional[str] = None
    summary: Optional[str] = None
    source_item_id: Optional[str] = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


EVENT_SCHEMA_FIELDS = [
    "event_id", "event_date", "collected_time", "title", "summary",
    "source_name", "source_url", "source_tier", "author", "topic", "event_name", "industry",
    "is_noise", "dedup_group_id", "dataset_id", "raw_payload",
    "agent_scored", "agent_classified", "agent_summarized",
    "agent_reasoning", "agent_confidence", "agent_risk_or_caveat", "agent_summary",
    "agent_image_insights", "selected_image_paths",
]


class NewsSourceAdapter:
    """Base adapter with rate-limiting built in."""

    def __init__(self, rate_limit: float = 10.0):
        self.rate_limit = rate_limit
        self._last_fetch: float = 0.0

    def fetch(self, since: Optional[datetime] = None, limit: Optional[int] = None) -> list[RawNewsItem]:
        raise NotImplementedError

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_fetch
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_fetch = time.time()
