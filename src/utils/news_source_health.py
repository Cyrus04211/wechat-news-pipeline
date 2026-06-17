"""Source health monitoring for the news collection pipeline.

Tracks consecutive failures per source and auto-disables sources
that exceed the failure threshold. Generates health reports.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_PATH = "data/news/logs/source_health.json"
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_AUTO_DISABLE = False


class SourceHealthTracker:
    """Tracks source health and auto-disables unreliable sources."""

    def __init__(
        self,
        health_path: str = DEFAULT_HEALTH_PATH,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        auto_disable: bool = DEFAULT_AUTO_DISABLE,
    ):
        self.health_path = health_path
        self.max_consecutive_failures = max_consecutive_failures
        self.auto_disable = auto_disable
        self._state: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.health_path):
            try:
                with open(self.health_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"sources": {}, "updated_at": ""}

    def _save(self) -> None:
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        Path(self.health_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.health_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def is_disabled(self, source_name: str) -> bool:
        """Check if a source is currently disabled."""
        if not self.auto_disable:
            return False
        src = self._state.get("sources", {}).get(source_name, {})
        return src.get("disabled", False)

    def record_success(self, source_name: str, item_count: int = 0) -> None:
        """Record a successful fetch from a source."""
        src = self._state.setdefault("sources", {}).setdefault(source_name, {})
        src["consecutive_failures"] = 0
        src["last_success"] = datetime.now(timezone.utc).isoformat()
        src["last_item_count"] = item_count
        src["total_successes"] = src.get("total_successes", 0) + 1
        if src.get("disabled"):
            src["disabled"] = False
            logger.info(f"[Health] Re-enabled source: {source_name}")
        self._save()

    def record_failure(self, source_name: str, error: str = "") -> bool:
        """Record a failed fetch. Returns True if the source was just auto-disabled."""
        src = self._state.setdefault("sources", {}).setdefault(source_name, {})
        src["consecutive_failures"] = src.get("consecutive_failures", 0) + 1
        src["last_failure"] = datetime.now(timezone.utc).isoformat()
        src["last_error"] = error[:200]
        src["total_failures"] = src.get("total_failures", 0) + 1

        failures = src["consecutive_failures"]
        just_disabled = False

        if self.auto_disable and failures >= self.max_consecutive_failures and not src.get("disabled"):
            src["disabled"] = True
            src["disabled_at"] = datetime.now(timezone.utc).isoformat()
            just_disabled = True
            logger.warning(
                f"[Health] AUTO-DISABLED source '{source_name}' "
                f"after {failures} consecutive failures. "
                f"Last error: {error[:100]}"
            )

        self._save()
        return just_disabled

    def get_status(self, source_name: str) -> dict:
        """Get health status for a source."""
        return self._state.get("sources", {}).get(source_name, {})

    def get_all_status(self) -> dict:
        """Get health status for all sources."""
        return self._state.get("sources", {})

    def get_disabled_sources(self) -> list[str]:
        """List all currently disabled sources."""
        if not self.auto_disable:
            return []
        return [
            name for name, info in self._state.get("sources", {}).items()
            if info.get("disabled", False)
        ]

    def reset_source(self, source_name: str) -> None:
        """Manually reset a source's health status."""
        self._state.setdefault("sources", {}).pop(source_name, None)
        self._save()
        logger.info(f"[Health] Reset health status for: {source_name}")

    def get_health_report(self) -> str:
        """Generate a human-readable health report."""
        sources = self._state.get("sources", {})
        if not sources:
            return "No source health data available."

        lines = ["Source Health Report:", "-" * 70]
        for name, info in sorted(sources.items()):
            status = "DISABLED" if info.get("disabled") else "OK"
            fails = info.get("consecutive_failures", 0)
            total_ok = info.get("total_successes", 0)
            total_fail = info.get("total_failures", 0)
            last_ok = str(info.get("last_success", "-"))[:19]
            lines.append(
                f"  [{status:<8}] {name:<20} "
                f"ok={total_ok} fail={total_fail} consec_fail={fails} "
                f"last_ok={last_ok}"
            )
        disabled = self.get_disabled_sources()
        if disabled:
            lines.append(f"\n  Disabled sources: {', '.join(disabled)}")
        return "\n".join(lines)
