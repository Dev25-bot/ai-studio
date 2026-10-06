"""Model registry: a factory that loads models on demand and caches them (LRU).

Backends
--------
- ``hf``     : download/load from the Hugging Face Hub by model id.
- ``local``  : load from a local checkpoint directory (same loaders, path instead of id).
- ``remote`` : no local weights; tasks call an OpenAI-compatible endpoint (text tasks only).
"""
from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import torch

from core.config import AppConfig
from core.device import diffusion_dtype, pipeline_device, resolve_device
from core.safety import UserFacingError
from models.remote import RemoteClient

log = logging.getLogger("ai_studio.registry")

REMOTE_CAPABLE = {"summarization", "sentiment", "next_word", "translation"}


def _load_image_gen(model: str, device: str, amp: bool) -> Any:
    from diffusers import AutoPipelineForText2Image
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()  # sd-turbo ships without a safety checker; prompts are
    diffusers_logging.disable_progress_bar()  # screened by core.safety.check_image_prompt instead

    dtype = diffusion_dtype(device, amp)
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    pipe = None
    if dtype == torch.float16:
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(model, variant="fp16", **kwargs)
        except Exception:  # repo has no fp16 variant → fall back to default weights
            pipe = None
    if pipe is None:
        pipe = AutoPipelineForText2Image.from_pretrained(model, **kwargs)
    pipe = pipe.to(device)
    if device == "mps":
        pipe.enable_attention_slicing()  # lowers peak memory on 8–16 GB Macs
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_causal_lm(model: str, device: str, amp: bool) -> Any:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model).to(device).eval()
    return tok, lm


def _hf_pipeline(task: str, **extra: Any) -> Callable[[str, str, bool], Any]:
    def load(model: str, device: str, amp: bool) -> Any:
        from transformers import pipeline

        return pipeline(task, model=model, device=pipeline_device(device), **extra)

    return load


LOADERS: dict[str, Callable[[str, str, bool], Any]] = {
    "image_gen": _load_image_gen,
    "summarization": _hf_pipeline("summarization"),
    "sentiment": _hf_pipeline("text-classification", top_k=None),
    "next_word": _load_causal_lm,
    "translation": _hf_pipeline("translation"),
    "ner": _hf_pipeline("token-classification", aggregation_strategy="simple"),
    "qa": _hf_pipeline("question-answering"),
    "image_classify": _hf_pipeline("image-classification"),
}


class ModelRegistry:
    """Thread-safe, LRU-bounded cache of loaded models keyed by (task, model)."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.runtime.get("device", "auto"))
        self.amp = bool(cfg.runtime.get("amp", True))
        self.max_loaded = max(1, int(cfg.runtime.get("max_loaded_models", 3)))
        self._cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._lock = threading.RLock()
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        r = cfg.remote
        self.remote = RemoteClient(r.get("base_url", ""), r.get("api_key_env", "OPENAI_API_KEY"),
                                   r.get("model", ""))

    def backend(self, task: str) -> str:
        return self.cfg.tasks[task].backend

    def loaded(self) -> list[str]:
        with self._lock:
            return [f"{t}:{m}" for t, m in self._cache]

    def get(self, task: str, model: str | None = None) -> Any:
        """Return a ready-to-use model object for ``task`` (loading and caching it if needed)."""
        if task not in LOADERS:
            raise UserFacingError(f"Unknown task '{task}'")
        tc = self.cfg.tasks.get(task)
        backend = tc.backend if tc else "hf"
        model = model or (tc.model if tc else "")
        if backend == "remote":
            if task not in REMOTE_CAPABLE:
                raise UserFacingError(f"Task '{task}' does not support the remote backend.")
            return self.remote
        if backend == "local" and not Path(model).exists():
            raise UserFacingError(f"Local checkpoint not found: {model}")

        key = (task, model)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        with key_lock:  # only one thread loads a given model
            with self._lock:
                if key in self._cache:
                    return self._cache[key]
            log.info("Loading %s model %s on %s", task, model, self.device)
            try:
                obj = LOADERS[task](model, self.device, self.amp)
            except OSError as exc:
                raise UserFacingError(
                    f"Could not load model '{model}'. Check the model id / internet connection. ({exc})"
                ) from exc
            with self._lock:
                self._cache[key] = obj
                while len(self._cache) > self.max_loaded:
                    old_key, _ = self._cache.popitem(last=False)
                    log.info("Evicting %s to stay under max_loaded_models=%d", old_key, self.max_loaded)
                    self._free()
            return obj

    def unload_task(self, task: str) -> None:
        """Drop every cached model for ``task`` and release its memory."""
        with self._lock:
            for key in [k for k in self._cache if k[0] == task]:
                del self._cache[key]
                log.info("Unloaded %s", key)
        self._free()

    def _free(self) -> None:
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
