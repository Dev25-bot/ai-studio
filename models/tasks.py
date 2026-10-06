"""Task runners. Each returns plain Python data plus a ``meta`` dict (latency, tokens, steps)
so the UI, the benchmark and the tests can share them."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from PIL import Image

from core.device import autocast
from core.safety import UserFacingError
from models.registry import ModelRegistry
from models.remote import RemoteClient


@dataclass
class Result:
    output: Any
    meta: dict[str, Any] = field(default_factory=dict)


def _count_tokens(pipe: Any, text: str) -> int:
    tok = getattr(pipe, "tokenizer", None)
    return len(tok(text)["input_ids"]) if tok is not None else len(text.split())


def _chunk_words(text: str, size: int = 600) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [text]


# ---------------------------------------------------------------- image generation
def generate_image(reg: ModelRegistry, prompt: str, model: str | None = None, negative_prompt: str = "",
                   steps: int = 2, guidance_scale: float = 0.0, width: int = 512, height: int = 512,
                   seed: int = 42) -> Result:
    pipe = reg.get("image_gen", model)
    gen_device = "cpu" if reg.device == "mps" else reg.device  # MPS generators are not reproducible
    generator = torch.Generator(gen_device).manual_seed(int(seed))
    t0 = time.perf_counter()
    with torch.inference_mode(), autocast(reg.device, reg.amp):
        kwargs: dict[str, Any] = dict(prompt=prompt, num_inference_steps=int(steps),
                                      guidance_scale=float(guidance_scale), width=int(width),
                                      height=int(height), generator=generator)
        if negative_prompt and guidance_scale > 1.0:
            kwargs["negative_prompt"] = negative_prompt
        image: Image.Image = pipe(**kwargs).images[0]
    dt = time.perf_counter() - t0
    return Result(image, {"latency_s": dt, "steps": int(steps), "s_per_step": dt / max(1, int(steps)),
                          "size": f"{width}×{height}", "seed": int(seed)})


# ---------------------------------------------------------------- summarization
MIN_SUMMARY_WORDS = 30


def summarize(reg: ModelRegistry, text: str, model: str | None = None, max_new_tokens: int = 130,
              min_new_tokens: int = 30, num_beams: int = 4) -> Result:
    n_words = len(text.split())
    if n_words < MIN_SUMMARY_WORDS:
        # Abstractive models hallucinate and repeat themselves on tiny inputs.
        raise UserFacingError(f"Text is only {n_words} words — paste at least {MIN_SUMMARY_WORDS} words to summarize.")
    obj = reg.get("summarization", model)
    t0 = time.perf_counter()
    if isinstance(obj, RemoteClient):
        summary = obj.chat("Summarize the user's text concisely in 2-4 sentences.", text, max_new_tokens)
        in_tok = len(text.split())
    else:
        in_tok = _count_tokens(obj, text)
        prefix = "summarize: " if "t5" in obj.model.name_or_path.lower() else ""
        parts = []
        beams = max(1, int(num_beams))
        # Checkpoint defaults like early_stopping/length_penalty only apply to beam search;
        # reset them for greedy decoding so transformers doesn't warn on every call.
        beam_kwargs: dict[str, Any] = {} if beams > 1 else {"early_stopping": False, "length_penalty": 1.0}
        for chunk in _chunk_words(text):
            out = obj(prefix + chunk, max_new_tokens=int(max_new_tokens),
                      min_new_tokens=min(int(min_new_tokens), int(max_new_tokens)),
                      num_beams=beams, truncation=True, do_sample=False, **beam_kwargs)
            parts.append(out[0]["summary_text"].strip())
        summary = " ".join(parts)
    dt = time.perf_counter() - t0
    ratio = len(summary.split()) / max(1, len(text.split()))
    return Result(summary, {"latency_s": dt, "input_tokens": in_tok,
                            "output_words": len(summary.split()), "length_ratio": round(ratio, 3)})


# ---------------------------------------------------------------- sentiment
def classify_sentiment(reg: ModelRegistry, text: str, model: str | None = None) -> Result:
    obj = reg.get("sentiment", model)
    t0 = time.perf_counter()
    if isinstance(obj, RemoteClient):
        ans = obj.chat("Classify sentiment. Reply with exactly POSITIVE or NEGATIVE.", text, 5, 0.0)
        label = "POSITIVE" if "POS" in ans.upper() else "NEGATIVE"
        scores = {label: 1.0, ("NEGATIVE" if label == "POSITIVE" else "POSITIVE"): 0.0}
    else:
        raw = obj(text, truncation=True)
        raw = raw[0] if raw and isinstance(raw[0], list) else raw
        scores = {d["label"].upper(): float(d["score"]) for d in raw}
    dt = time.perf_counter() - t0
    label = max(scores, key=scores.get)
    return Result({"label": label, "score": scores[label], "scores": scores}, {"latency_s": dt})


def classify_batch(reg: ModelRegistry, texts: list[str], model: str | None = None,
                   batch_size: int = 16) -> list[str]:
    """Batched sentiment labels (used by the benchmark)."""
    obj = reg.get("sentiment", model)
    if isinstance(obj, RemoteClient):
        return [classify_sentiment(reg, t).output["label"] for t in texts]
    outs = obj(texts, truncation=True, batch_size=batch_size)
    return [max(o, key=lambda d: d["score"])["label"].upper() for o in outs]


# ---------------------------------------------------------------- next-word prediction
def top_next_tokens(reg: ModelRegistry, prompt: str, model: str | None = None, k: int = 10) -> list[tuple[str, float]]:
    """Most likely next tokens with probabilities."""
    obj = reg.get("next_word", model)
    if isinstance(obj, RemoteClient):
        return []
    tok, lm = obj
    ids = tok(prompt, return_tensors="pt").to(reg.device)
    with torch.inference_mode():
        logits = lm(**ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    p, idx = probs.topk(k)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(p, idx)]


def stream_continuation(reg: ModelRegistry, prompt: str, model: str | None = None, max_new_tokens: int = 40,
                        temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                        seed: int = 42) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (generated_text_so_far, meta) as tokens stream in."""
    obj = reg.get("next_word", model)
    t0 = time.perf_counter()
    if isinstance(obj, RemoteClient):
        text = obj.chat("Continue the user's text naturally. Output only the continuation.",
                        prompt, int(max_new_tokens), float(temperature))
        yield text, {"latency_s": time.perf_counter() - t0, "new_tokens": len(text.split())}
        return

    from transformers import TextIteratorStreamer

    tok, lm = obj
    torch.manual_seed(int(seed))
    inputs = tok(prompt, return_tensors="pt").to(reg.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    do_sample = float(temperature) > 0
    gen_kwargs: dict[str, Any] = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens),
                                      do_sample=do_sample, pad_token_id=tok.eos_token_id)
    if do_sample:
        gen_kwargs.update(temperature=float(temperature), top_k=int(top_k), top_p=float(top_p))
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            with torch.inference_mode():
                lm.generate(**gen_kwargs)
        except BaseException as exc:  # surface errors from the worker thread
            errors.append(exc)
            streamer.end()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    text, n, first = "", 0, None
    for piece in streamer:
        if first is None:
            first = time.perf_counter() - t0
        text += piece
        n += 1
        elapsed = time.perf_counter() - t0
        yield text, {"latency_s": elapsed, "first_token_s": first, "new_tokens": n,
                     "tokens_per_s": n / max(1e-6, elapsed - (first or 0)) if n > 1 else 0.0}
    thread.join()
    if errors:
        raise errors[0]


# ---------------------------------------------------------------- optional tasks
def translate(reg: ModelRegistry, text: str, model: str | None = None, max_new_tokens: int = 256) -> Result:
    obj = reg.get("translation", model)
    t0 = time.perf_counter()
    if isinstance(obj, RemoteClient):
        out = obj.chat("Translate the user's English text to German. Output only the translation.",
                       text, int(max_new_tokens))
    else:
        out = obj(text, max_new_tokens=int(max_new_tokens), truncation=True)[0]["translation_text"]
    return Result(out, {"latency_s": time.perf_counter() - t0})


def extract_entities(reg: ModelRegistry, text: str, model: str | None = None) -> Result:
    pipe = reg.get("ner", model)
    t0 = time.perf_counter()
    ents = pipe(text)
    ents = [{"entity": e["entity_group"], "word": e["word"], "score": round(float(e["score"]), 4),
             "start": int(e["start"]), "end": int(e["end"])} for e in ents]
    return Result(ents, {"latency_s": time.perf_counter() - t0, "entities": len(ents)})


def answer_question(reg: ModelRegistry, question: str, context: str, model: str | None = None) -> Result:
    pipe = reg.get("qa", model)
    t0 = time.perf_counter()
    out = pipe(question=question, context=context)
    return Result({"answer": out["answer"], "score": float(out["score"]),
                   "start": int(out["start"]), "end": int(out["end"])},
                  {"latency_s": time.perf_counter() - t0})


def classify_image(reg: ModelRegistry, image: Image.Image, model: str | None = None, top_k: int = 5) -> Result:
    pipe = reg.get("image_classify", model)
    t0 = time.perf_counter()
    preds = pipe(image.convert("RGB"), top_k=int(top_k))
    return Result({p["label"]: float(p["score"]) for p in preds}, {"latency_s": time.perf_counter() - t0})
