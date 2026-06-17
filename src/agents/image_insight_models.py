"""Dataclasses for image insight analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageInsight:
    image_index: int = 0
    keep: bool = False
    explanation: str = ""
