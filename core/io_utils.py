"""Persist task outputs as downloadable artifacts (PNG / TXT / JSON)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image


def _stem(task: str, hint: str = "") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", hint.lower()).strip("-")[:40]
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    return f"{task}_{stamp}" + (f"_{slug}" if slug else "")


def save_image(img: Image.Image, out_dir: Path, task: str, hint: str = "") -> str:
    d = out_dir / task
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_stem(task, hint)}.png"
    img.save(p)
    return str(p)


def save_text(text: str, out_dir: Path, task: str, hint: str = "") -> str:
    d = out_dir / task
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_stem(task, hint)}.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


def save_json(data: Any, out_dir: Path, task: str, hint: str = "") -> str:
    d = out_dir / task
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_stem(task, hint)}.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(p)
