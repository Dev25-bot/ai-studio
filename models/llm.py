"""Local chat engine on Apple Silicon (MLX).

One class, two backends:
- ``mlx_lm``  for text-only checkpoints (fastest),
- ``mlx_vlm`` for vision-language checkpoints (text + images in the same chat).

``ChatEngine.stream(messages, images)`` yields ``Chunk``s (text delta + running stats), so the UI,
the evaluation script and the tests share exactly the same generation path.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

log = logging.getLogger("ai_studio.llm")

DEFAULT_SYSTEM_PROMPT = (
    "You are AI Studio, a fast, precise assistant running locally on the user's Mac.\n"
    "- Be correct first, then concise. Answer simple questions directly in natural sentences.\n"
    "- For maths, logic, counting or trick questions only: read the question literally, reason step by step, "
    "double-check against the question, and end with a final line \"Final answer: <answer>\".\n"
    "- Code: working, idiomatic code in a fenced block with the language tag; short explanations.\n"
    "- When the user attaches a document, treat it as the source of truth for summaries, translations, "
    "extraction and questions.\n"
    "- If you are unsure or the information may be outdated, say so instead of guessing.\n"
    "- Use Markdown (headings, lists, tables) when it makes the answer easier to read."
)


@dataclass
class Chunk:
    text: str                     # new text since the previous chunk
    prompt_tokens: int = 0
    generated_tokens: int = 0
    tokens_per_s: float = 0.0
    first_token_s: float | None = None
    elapsed_s: float = 0.0
    finish_reason: str | None = None


@dataclass
class EngineConfig:
    model: str
    vision: bool = False
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    context_tokens: int = 6144          # prompt budget; older turns are trimmed beyond this
    keepalive_s: float = 15.0           # touch the weights this often while idle (0 = off); see _keepalive
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    extra: dict[str, Any] = field(default_factory=dict)


class ChatEngine:
    """Loads one MLX checkpoint lazily and streams chat completions. Thread-safe (one generation at a time)."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._proc: Any = None          # tokenizer (mlx_lm) or processor (mlx_vlm)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._caches: dict[str, Any] = {}  # mlx_vlm KV-prefix + vision-feature caches (reused across turns)

    # ------------------------------------------------------------------ lifecycle
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, warm: bool = False) -> None:
        """Load weights; ``warm`` also runs a 1-token generation so GPU kernels are compiled up front."""
        if self._model is not None:
            return
        t0 = time.perf_counter()
        if self.cfg.vision:
            from mlx_vlm import load
        else:
            from mlx_lm import load
        self._model, self._proc = load(self.cfg.model)
        self._reset_caches()
        if warm:
            for _ in self._raw_stream(self._prompt([{"role": "user", "content": "Hi"}], 0), [], 1, 0.0):
                pass
            self._reset_caches()  # don't keep the warm-up prompt as a cache prefix
        log.info("Loaded %s in %.1fs", self.cfg.model, time.perf_counter() - t0)

    def start_keepalive(self) -> None:
        """Keep the weights "recently used" so macOS doesn't swap them out while the app sits idle.

        On 8 GB Macs an idle model gets paged out by other apps, and the next reply then waits seconds for the
        weights to come back. Reading every weight once (one reduction, ~30 ms of GPU time) every few seconds
        prevents that. Wiring the memory with mx.set_wired_limit was measured and did not help.
        """
        if self.cfg.keepalive_s <= 0 or getattr(self, "_keepalive_thread", None):
            return

        def loop() -> None:
            import mlx.core as mx
            from mlx.utils import tree_flatten

            while True:
                time.sleep(self.cfg.keepalive_s)
                model = self._model
                if model is None or not self._lock.acquire(blocking=False):
                    continue  # unloaded, or a reply is being generated (that touches the weights anyway)
                try:
                    mx.eval([mx.sum(v) for _, v in tree_flatten(model.parameters())])
                except Exception as exc:  # pragma: no cover - never let the keep-alive kill the app
                    log.debug("keepalive skipped: %s", exc)
                finally:
                    self._lock.release()

        self._keepalive_thread = threading.Thread(target=loop, daemon=True, name="llm-keepalive")
        self._keepalive_thread.start()

    def _reset_caches(self) -> None:
        if self.cfg.vision:
            from mlx_vlm.generate.common import PromptCacheState
            from mlx_vlm.vision_cache import VisionFeatureCache

            self._caches = {"prompt_cache_state": PromptCacheState(), "vision_cache": VisionFeatureCache()}

    def unload(self) -> None:
        self._model = self._proc = None
        self._caches = {}
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # pragma: no cover - best effort
            pass

    def stop(self) -> None:
        """Ask the running generation to finish early (UI Stop button)."""
        self._stop.set()

    # ------------------------------------------------------------------ prompting
    @property
    def tokenizer(self) -> Any:
        return getattr(self._proc, "tokenizer", self._proc)

    def count_tokens(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode(text))

    def _trim(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Keep the system prompt + the newest turns that fit the context budget."""
        system = [m for m in messages if m["role"] == "system"]
        turns = [m for m in messages if m["role"] != "system"]
        budget = self.cfg.context_tokens - sum(self.count_tokens(m["content"]) for m in system)
        kept: list[dict[str, str]] = []
        for m in reversed(turns):
            cost = self.count_tokens(m["content"]) + 8
            if kept and cost > budget:
                break
            kept.insert(0, m)
            budget -= cost
        return system + kept

    def build_messages(self, history: list[dict[str, str]], system_prompt: str | None = None) -> list[dict[str, str]]:
        sp = self.cfg.system_prompt if system_prompt is None else system_prompt
        msgs = ([{"role": "system", "content": sp}] if sp else []) + history
        return self._trim(msgs)

    def _prompt(self, messages: list[dict[str, str]], n_images: int) -> str:
        if self.cfg.vision:
            from mlx_vlm.prompt_utils import apply_chat_template

            return apply_chat_template(self._proc, self._model.config, messages, num_images=n_images)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ generation
    def stream(self, messages: list[dict[str, str]], images: list[str] | None = None,
               max_tokens: int | None = None, temperature: float | None = None) -> Iterator[Chunk]:
        """Yield text deltas for the assistant reply to ``messages`` (already includes the system prompt)."""
        images = images or []
        if images and not self.cfg.vision:
            raise ValueError(f"{self.cfg.model} is a text-only model and cannot read images.")
        self.load()
        max_tokens = int(max_tokens or self.cfg.max_tokens)
        temp = self.cfg.temperature if temperature is None else float(temperature)

        with self._lock:
            self._stop.clear()
            prompt = self._prompt(messages, len(images))
            t0 = time.perf_counter()
            first: float | None = None
            last = None
            for r in self._raw_stream(prompt, images, max_tokens, temp):
                if first is None:
                    first = time.perf_counter() - t0
                last = r
                yield Chunk(r.text, r.prompt_tokens, r.generation_tokens, float(r.generation_tps or 0.0),
                            first, time.perf_counter() - t0, getattr(r, "finish_reason", None))
                if self._stop.is_set():
                    yield Chunk("", last.prompt_tokens, last.generation_tokens, float(last.generation_tps or 0.0),
                                first, time.perf_counter() - t0, "stopped")
                    return

    def _raw_stream(self, prompt: str, images: list[str], max_tokens: int, temp: float) -> Iterator[Any]:
        if self.cfg.vision:
            from mlx_vlm import stream_generate

            yield from stream_generate(self._model, self._proc, prompt, image=images or None,
                                       max_tokens=max_tokens, temperature=temp, top_p=self.cfg.top_p,
                                       repetition_penalty=self.cfg.repetition_penalty,
                                       **self._caches)  # follow-up turns only prefill the new tokens
            return
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        yield from stream_generate(self._model, self._proc, prompt, max_tokens=max_tokens,
                                   sampler=make_sampler(temp=temp, top_p=self.cfg.top_p if temp > 0 else 0.0),
                                   logits_processors=make_logits_processors(
                                       repetition_penalty=self.cfg.repetition_penalty))

    def complete(self, messages: list[dict[str, str]], images: list[str] | None = None, **kw: Any) -> tuple[str, Chunk]:
        """Non-streaming helper: full text + final stats."""
        text, last = "", Chunk("")
        for c in self.stream(messages, images, **kw):
            text += c.text
            last = c
        return text, last
