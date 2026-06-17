"""Manual review workflow for research evidence cards."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.utils.news_storage import ensure_dir

VALID_STATUSES = {"new", "watch", "verified", "rejected", "report_ready"}
UTC = timezone.utc


class ReviewWorkflow:
    def __init__(self, state_path: str = "data/news/research/review_state/review_state.json"):
        self.state_path = state_path

    def load_states(self) -> dict[str, dict[str, Any]]:
        if not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data.get("events", data) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def save_states(self, states: dict[str, dict[str, Any]]) -> str:
        ensure_dir(os.path.dirname(self.state_path))
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            "events": states,
        }
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return self.state_path

    def set_status(self, event_id: str, status: str, note: str = "") -> dict[str, Any]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of {VALID_STATUSES}")
        states = self.load_states()
        entry = states.get(event_id, {})
        entry["status"] = status
        if note:
            entry["note"] = note
        entry["updated_at"] = datetime.now(UTC).isoformat()
        states[event_id] = entry
        self.save_states(states)
        return entry

    def list_events(
        self,
        status: Optional[str] = None,
        evidence_cards_path: str = "data/news/research/evidence_cards/evidence_cards.csv",
    ) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        if os.path.exists(evidence_cards_path):
            df = pd.read_csv(evidence_cards_path)
            cards = df.to_dict(orient="records")
        elif os.path.exists(evidence_cards_path.replace(".csv", ".parquet")):
            df = pd.read_parquet(evidence_cards_path.replace(".csv", ".parquet"))
            cards = df.to_dict(orient="records")

        states = self.load_states()
        results = []
        for card in cards:
            eid = str(card.get("event_id", ""))
            review = states.get(eid, {})
            card_status = review.get("status", card.get("reviewer_status", "new"))
            if status and card_status != status:
                continue
            results.append({
                "event_id": eid,
                "title": card.get("title", ""),
                "industry": card.get("industry", ""),
                "etf": card.get("etf", ""),
                "status": card_status,
                "note": review.get("note", card.get("reviewer_notes", "")),
                "source_name": card.get("source_name", ""),
            })
        results.sort(key=lambda x: (x["event_id"],))
        return results

    def export_watchlist(
        self,
        output_path: str = "data/news/research/review_state/watchlist.csv",
    ) -> str:
        items = self.list_events(status="watch")
        items += [i for i in self.list_events(status="new") if i not in items]
        ensure_dir(os.path.dirname(output_path))
        df = pd.DataFrame(items)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path

    def merge_into_cards(self, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        states = self.load_states()
        for card in cards:
            eid = str(card.get("event_id", ""))
            if eid in states:
                card["reviewer_status"] = states[eid].get("status", card.get("reviewer_status", "new"))
                if states[eid].get("note"):
                    card["reviewer_notes"] = states[eid]["note"]
        return cards
