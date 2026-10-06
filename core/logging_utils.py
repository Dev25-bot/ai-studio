"""Structured JSON-lines logging with optional Weights & Biases mirroring."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from core.config import ROOT

_logger = logging.getLogger("ai_studio")
_lock = threading.Lock()


def _trunc(value: Any, n: int = 500) -> Any:
    if isinstance(value, str) and len(value) > n:
        return value[:n] + f"…(+{len(value) - n} chars)"
    return value


class EventLogger:
    """Writes one JSON object per inference event to a .jsonl file."""

    def __init__(self, path: str, level: str = "INFO", log_payloads: bool = True, wandb: bool = False):
        p = Path(path)
        self.path = p if p.is_absolute() else ROOT / p
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_payloads = log_payloads
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._wandb = None
        if wandb and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb as _wb  # type: ignore

                self._wandb = _wb.init(project="multi-task-ai-studio", reinit=True)
            except Exception as exc:  # pragma: no cover - optional dependency
                _logger.warning("W&B disabled: %s", exc)

    def log(self, task: str, model: str, status: str, latency_s: float,
            inputs: dict[str, Any] | None = None, outputs: dict[str, Any] | None = None,
            extra: dict[str, Any] | None = None) -> None:
        event: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task": task, "model": model, "status": status,
            "latency_s": round(latency_s, 4),
        }
        if self.log_payloads:
            event["inputs"] = {k: _trunc(v) for k, v in (inputs or {}).items()}
            event["outputs"] = {k: _trunc(v) for k, v in (outputs or {}).items()}
        if extra:
            event.update(extra)
        line = json.dumps(event, default=str, ensure_ascii=False)
        with _lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        _logger.info("%s %s %s %.3fs", task, model, status, latency_s)
        if self._wandb is not None:
            self._wandb.log({f"{task}/latency_s": latency_s})
