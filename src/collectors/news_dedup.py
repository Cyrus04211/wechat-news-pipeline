from datetime import datetime, timedelta
from typing import Optional

from src.utils.news_text import (
    compute_char_ngram_similarity,
    compute_event_id,
    compute_title_similarity,
)


class DeduplicationEngine:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self._existing_event_ids: set[str] = set()
        self._existing_dedup_group_ids: set[str] = set()
        self._history: list[dict] = []

    def load_existing(self, event_ids: set[str], dedup_group_ids: Optional[set[str]] = None) -> None:
        self._existing_event_ids = event_ids
        self._existing_dedup_group_ids = dedup_group_ids or set()

    def load_history(self, events: list[dict]) -> None:
        self._history = events

    def add_to_history(self, item: dict) -> None:
        self._history.append(item)

    def check(
        self,
        title: str,
        source_name: str,
        event_date: datetime,
        industry: str,
        dedup_group_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        # Check by event_id against existing store
        eid = compute_event_id(title, source_name, event_date.isoformat(), industry)
        if dedup_group_id and dedup_group_id in self._existing_dedup_group_ids:
            return True, dedup_group_id
        if eid in self._existing_event_ids:
            return True, eid

        # Check by similarity against current batch history
        for prev in self._history:
            prev_industry = prev.get("industry") or ""
            if industry and prev_industry and prev_industry != industry:
                continue
            prev_time = prev.get("event_date")
            if isinstance(prev_time, datetime) and abs((event_date - prev_time)) > timedelta(hours=24):
                continue
            word_sim = compute_title_similarity(title, prev.get("title", ""))
            char_sim = compute_char_ngram_similarity(title, prev.get("title", ""), n=3)
            sim = max(word_sim, char_sim)
            if sim >= self.similarity_threshold:
                return True, prev.get("dedup_group_id") or prev.get("event_id")

        return False, eid


def compute_dedup_group_id(title: str, industry: str, event_date: datetime) -> str:
    from src.utils.news_text import normalize_title_for_dedup
    import hashlib
    raw = f"{normalize_title_for_dedup(title)}|{industry}|{event_date.strftime('%Y-%m-%d')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
