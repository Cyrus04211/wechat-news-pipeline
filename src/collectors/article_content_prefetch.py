"""Backward-compatible re-export of article media prefetch."""

from src.collectors.article_media_prefetch import (
    attach_article_content_to_events,
    attach_article_media_to_events,
)

__all__ = ["attach_article_content_to_events", "attach_article_media_to_events"]
