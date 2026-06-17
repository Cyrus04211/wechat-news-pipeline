from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def find_config_path(name: str, override: Optional[str] = None) -> Path:
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.extend([
        Path(name),
        Path("config") / name,
        _CONFIG_DIR / name,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_yaml_config(name: str, override: Optional[str] = None) -> dict[str, Any]:
    path = find_config_path(name, override)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_pipeline_config(path: Optional[str] = None) -> dict[str, Any]:
    return load_yaml_config("news_runtime.yaml", override=path)


def load_runtime_config(path: Optional[str] = None) -> dict[str, Any]:
    return load_pipeline_config(path).get("runtime", {})


def load_llm_config(path: Optional[str] = None) -> dict[str, Any]:
    return load_pipeline_config(path).get("llm", {})


def load_timeliness_config(path: Optional[str] = None) -> dict[str, Any]:
    return load_pipeline_config(path).get("timeliness", {})


def load_sources_config(path: Optional[str] = None) -> dict[str, Any]:
    return load_yaml_config("news_sources.yaml", override=path).get("sources", {})


def load_collection_profiles_config(path: Optional[str] = None) -> dict[str, Any]:
    return load_pipeline_config(path).get("collection_profiles", {})


def load_schedule_config(path: Optional[str] = None) -> dict[str, Any]:
    return load_pipeline_config(path).get("schedule", {})


def load_research_config(path: Optional[str] = None) -> dict[str, Any]:
    return load_yaml_config("research_runtime.yaml", override=path)
