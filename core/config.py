"""Config loading: YAML → typed dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "app.yaml"

TASK_ORDER = [
    "image_gen", "summarization", "sentiment", "next_word",
    "translation", "ner", "qa", "image_classify",
]
VALID_BACKENDS = {"hf", "local", "remote"}


class ConfigError(ValueError):
    """Raised when the YAML config is invalid."""


@dataclass
class TaskConfig:
    name: str
    enabled: bool = True
    label: str = ""
    backend: str = "hf"
    model: str = ""
    alternatives: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def model_choices(self) -> list[str]:
        """Default model first, then alternatives (deduplicated)."""
        return list(dict.fromkeys([self.model, *self.alternatives]))


@dataclass
class AppConfig:
    app: dict[str, Any]
    runtime: dict[str, Any]
    limits: dict[str, Any]
    logging: dict[str, Any]
    remote: dict[str, Any]
    eval: dict[str, Any]
    tasks: dict[str, TaskConfig]
    chat: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled_tasks(self) -> list[TaskConfig]:
        return [self.tasks[n] for n in TASK_ORDER if n in self.tasks and self.tasks[n].enabled]

    @property
    def outputs_dir(self) -> Path:
        p = Path(self.app.get("outputs_dir", "outputs"))
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the app config from YAML."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    tasks: dict[str, TaskConfig] = {}
    for name, spec in (raw.get("tasks") or {}).items():
        if name not in TASK_ORDER:
            raise ConfigError(f"Unknown task '{name}'. Valid: {TASK_ORDER}")
        spec = spec or {}
        tc = TaskConfig(
            name=name,
            enabled=bool(spec.get("enabled", True)),
            label=spec.get("label", name.replace("_", " ").title()),
            backend=spec.get("backend", "hf"),
            model=spec.get("model", ""),
            alternatives=list(spec.get("alternatives") or []),
            params=dict(spec.get("params") or {}),
        )
        if tc.backend not in VALID_BACKENDS:
            raise ConfigError(f"Task '{name}': backend must be one of {sorted(VALID_BACKENDS)}")
        if tc.enabled and not tc.model and tc.backend != "remote":
            raise ConfigError(f"Task '{name}' is enabled but has no model")
        tasks[name] = tc

    return AppConfig(
        app=raw.get("app") or {},
        runtime=raw.get("runtime") or {},
        limits=raw.get("limits") or {},
        logging=raw.get("logging") or {},
        remote=raw.get("remote") or {},
        eval=raw.get("eval") or {},
        tasks=tasks,
        chat=raw.get("chat") or {},
    )
