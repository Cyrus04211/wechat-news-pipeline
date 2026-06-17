from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_raw_jsonl(
    items: list[dict[str, Any]],
    source_name: str,
    event_date: datetime,
    raw_dir: str,
) -> str:
    date_str = event_date.strftime("%Y-%m-%d")
    dir_path = ensure_dir(os.path.join(raw_dir, source_name))
    file_path = os.path.join(dir_path, f"{date_str}.jsonl")
    with open(file_path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    logger.debug(f"Wrote {len(items)} raw items to {file_path}")
    return file_path


def _serialize_raw_payload(val: Any) -> Optional[str]:
    """Convert raw_payload to a JSON string for parquet storage."""
    if val is None:
        return None
    if isinstance(val, (str, bytes)):
        return str(val)
    try:
        return json.dumps(val, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(val)


def write_events_parquet(
    events: list[dict[str, Any]],
    output_path: str,
    existing_path: Optional[str] = None,
) -> str:
    if not events:
        logger.warning("No events to write to parquet")
        return output_path

    # Normalize fields to avoid PyArrow schema issues
    normalized = []
    for event in events:
        e = dict(event)
        if "raw_payload" in e:
            e["raw_payload"] = _serialize_raw_payload(e["raw_payload"])
        for key in ("keywords_by_level",):
            if isinstance(e.get(key), dict) and not e[key]:
                e[key] = "{}"
        if isinstance(e.get("cross_industry_effects"), list):
            e["cross_industry_effects"] = json.dumps(e["cross_industry_effects"], ensure_ascii=False)
        elif e.get("cross_industry_effects") is None:
            e["cross_industry_effects"] = ""
        normalized.append(e)

    # Merge with existing events if available
    if existing_path and existing_path != output_path and os.path.exists(existing_path):
        try:
            existing = pd.read_parquet(existing_path)
            new_ids = {e["event_id"] for e in normalized if e.get("event_id")}
            existing = existing[~existing["event_id"].isin(new_ids)]
            all_df = pd.concat([existing, pd.DataFrame(normalized)], ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to merge with existing parquet: {e}")
            all_df = pd.DataFrame(normalized)
    elif os.path.exists(output_path):
        try:
            existing = pd.read_parquet(output_path)
            new_ids = {e["event_id"] for e in normalized if e.get("event_id")}
            existing = existing[~existing["event_id"].isin(new_ids)]
            all_df = pd.concat([existing, pd.DataFrame(normalized)], ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to merge with existing parquet: {e}")
            all_df = pd.DataFrame(normalized)
    else:
        all_df = pd.DataFrame(normalized)

    ensure_dir(os.path.dirname(output_path))
    all_df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(all_df)} events ({len(normalized)} new, {len(all_df) - len(normalized)} existing) to {output_path}")
    return output_path


def write_records_parquet(
    records: list[dict[str, Any]],
    output_path: str,
    existing_path: Optional[str] = None,
) -> str:
    return write_events_parquet(records, output_path, existing_path=existing_path)


def write_events_csv(
    events: list[dict[str, Any]],
    output_path: str,
    existing_path: Optional[str] = None,
) -> str:
    if not events:
        logger.warning("No events to write to csv")
        return output_path

    # Merge with existing events
    source_path = existing_path or output_path
    if os.path.exists(source_path):
        try:
            existing = pd.read_csv(source_path)
            new_ids = {e["event_id"] for e in events if e.get("event_id")}
            if "event_id" in existing.columns:
                existing = existing[~existing["event_id"].isin(new_ids)]
            df = pd.concat([existing, pd.DataFrame(events)], ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to merge with existing csv: {e}")
            df = pd.DataFrame(events)
    else:
        df = pd.DataFrame(events)

    df = _normalize_event_columns(df)
    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote {len(df)} events to {output_path}")
    return output_path


def _normalize_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure event CSVs include the canonical schema columns."""
    try:
        from src.sources.news_base import EVENT_SCHEMA_FIELDS
    except Exception:
        return df

    for field in EVENT_SCHEMA_FIELDS:
        if field not in df.columns:
            df[field] = ""
    ordered = [field for field in EVENT_SCHEMA_FIELDS if field in df.columns]
    extras = [col for col in df.columns if col not in ordered]
    return df[ordered + extras]


def write_records_csv(
    records: list[dict[str, Any]],
    output_path: str,
    existing_path: Optional[str] = None,
) -> str:
    return write_events_csv(records, output_path, existing_path=existing_path)


def read_recent_events(parquet_path: str) -> pd.DataFrame:
    if not os.path.exists(parquet_path):
        return pd.DataFrame()
    df = pd.read_parquet(parquet_path)
    if "is_noise" in df.columns:
        df = df[df["is_noise"] == False]
    if "event_date" in df.columns:
        df = df.sort_values("event_date", ascending=False)
    return df


def load_existing_event_ids(parquet_path: str) -> set[str]:
    if not os.path.exists(parquet_path):
        return set()
    try:
        df = pd.read_parquet(parquet_path)
        return set(df["event_id"].dropna().tolist())
    except Exception as e:
        logger.warning(f"Failed to load existing events: {e}")
        return set()


def load_existing_dedup_keys(parquet_path: str) -> tuple[set[str], set[str]]:
    if not os.path.exists(parquet_path):
        return set(), set()
    try:
        df = pd.read_parquet(parquet_path)
        event_ids = set(df["event_id"].dropna().tolist()) if "event_id" in df.columns else set()
        dedup_group_ids = set(df["dedup_group_id"].dropna().tolist()) if "dedup_group_id" in df.columns else set()
        return event_ids, dedup_group_ids
    except Exception as e:
        logger.warning(f"Failed to load existing dedup keys: {e}")
        return set(), set()


def load_existing_events(parquet_path: str) -> list[dict]:
    """Load ALL existing event records from parquet, returning as list of dicts."""
    if not os.path.exists(parquet_path):
        return []
    try:
        df = pd.read_parquet(parquet_path)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"Failed to load existing events: {e}")
        return []


def load_recent_titles(
    parquet_path: str,
    hours: int = 48,
) -> list[dict]:
    if not os.path.exists(parquet_path):
        return []
    try:
        df = pd.read_parquet(parquet_path)
        time_col = "event_date"
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], format="mixed", utc=True)
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
            df = df[df[time_col] >= cutoff]
        cols = [col for col in ["title", "topic", "event_date", "dedup_group_id", "event_id"] if col in df.columns]
        return df[cols].to_dict("records")
    except Exception:
        return []
