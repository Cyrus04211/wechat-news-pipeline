"""Shared helpers for batched LLM image insight analysis."""

from __future__ import annotations

import json
import re
from typing import Any

from src.research.article_summary import parse_image_insights


def merge_image_insight_payloads(
    existing_raw: Any,
    new_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge image insight records by global image_index."""
    merged = {entry["image_index"]: entry for entry in parse_image_insights(existing_raw)}
    for entry in new_entries:
        try:
            image_index = int(entry.get("image_index", 0))
        except (TypeError, ValueError):
            continue
        merged[image_index] = {
            "image_index": image_index,
            "keep": bool(entry.get("keep", False)),
            "explanation": str(entry.get("explanation", "") or "").strip(),
        }
    return [merged[index] for index in sorted(merged)]


def missing_image_insight_indices(
    image_paths: list[str],
    existing_raw: Any,
    *,
    max_images: int = 0,
) -> list[int]:
    """Return image indices that still need LLM insight text."""
    if max_images and max_images > 0:
        limit = min(len(image_paths), int(max_images))
    else:
        limit = len(image_paths)
    if limit <= 0:
        return []

    existing = parse_image_insights(existing_raw)
    covered = {
        entry["image_index"]
        for entry in existing
        if str(entry.get("explanation", "") or "").strip()
    }
    return [index for index in range(limit) if index not in covered]


def group_indices_into_batches(indices: list[int], batch_size: int) -> list[list[int]]:
    """Group indices into API-sized batches while preserving ascending order."""
    if not indices:
        return []
    size = max(1, int(batch_size))
    batches: list[list[int]] = []
    current: list[int] = []
    for index in sorted(indices):
        if current and index != current[-1] + 1:
            batches.append(current)
            current = []
        current.append(index)
        if len(current) >= size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def dump_image_insights(entries: list[dict[str, Any]]) -> str:
    return json.dumps(entries, ensure_ascii=False)


def extract_json_array(raw_text: str) -> list[Any] | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1)
        text = re.sub(r"\s*```\s*$", "", text, count=1)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return None


def parse_image_insight_response(raw_text: str, item_id: str) -> list[dict[str, Any]]:
    data = extract_json_array(raw_text)
    if not isinstance(data, list):
        return []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("item_id", "") or "") not in ("", item_id):
            continue
        insights = entry.get("image_insights") or []
        if not isinstance(insights, list):
            return []
        normalized: list[dict[str, Any]] = []
        for insight in insights:
            if not isinstance(insight, dict):
                continue
            try:
                image_index = int(insight.get("image_index", 0))
            except (TypeError, ValueError):
                image_index = 0
            normalized.append({
                "image_index": image_index,
                "keep": bool(insight.get("keep", False)),
                "explanation": str(insight.get("explanation", "") or "").strip(),
            })
        return normalized
    return []
