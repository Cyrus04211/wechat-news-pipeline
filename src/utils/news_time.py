from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc
CST = timezone(timedelta(hours=8))

_TIMELINESS_OVERRIDE: int | None = None
_TIMELINESS_WINDOWS = {1: 4, 2: 24, 3: 24, 4: 48}


def set_timeliness_override(hours: int | None) -> None:
    """Override timeliness window for all tiers (used for full-range collection)."""
    global _TIMELINESS_OVERRIDE
    _TIMELINESS_OVERRIDE = hours


def configure_timeliness_windows(windows: dict[int, int] | dict[str, int] | None) -> None:
    global _TIMELINESS_WINDOWS
    if not windows:
        _TIMELINESS_WINDOWS = {1: 4, 2: 24, 3: 24, 4: 48}
        return

    normalized = {}
    for tier, hours in windows.items():
        try:
            normalized[int(tier)] = int(hours)
        except (TypeError, ValueError):
            continue
    if normalized:
        _TIMELINESS_WINDOWS = normalized


def utc_now() -> datetime:
    return datetime.now(UTC)


def cst_now() -> datetime:
    return datetime.now(CST)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def is_within_window(
    event_time: datetime,
    reference_time: datetime | None = None,
    hours: int = 4,
) -> bool:
    ref = reference_time or utc_now()
    event_utc = ensure_utc(event_time)
    ref_utc = ensure_utc(ref)
    return (ref_utc - event_utc) <= timedelta(hours=hours)


def timeliness_window_hours(tier: int) -> int:
    if _TIMELINESS_OVERRIDE is not None:
        return _TIMELINESS_OVERRIDE
    return _TIMELINESS_WINDOWS.get(tier, 24)


def timeliness_decay(
    event_time: datetime,
    reference_time: datetime | None = None,
    tier: int = 2,
) -> float:
    """Gradual timeliness decay multiplier in [0.0, 1.0].

    0–100% of tier window → 1.0 (no penalty)
    100%–200% of window  → linear 1.0 → 0.3
    >200%                 → 0.0 (drop)
    """
    ref = reference_time or utc_now()
    event_utc = ensure_utc(event_time)
    ref_utc = ensure_utc(ref)

    age_hours = (ref_utc - event_utc).total_seconds() / 3600
    if age_hours < 0:
        return 1.0

    window_hours = float(timeliness_window_hours(tier))

    if age_hours <= window_hours:
        return 1.0
    elif age_hours <= window_hours * 2:
        overshoot = (age_hours - window_hours) / window_hours
        return round(1.0 - 0.7 * overshoot, 2)
    else:
        return 0.0
