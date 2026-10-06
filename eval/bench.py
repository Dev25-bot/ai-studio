"""Benchmark mode: run small test sets through each task and print a metrics table.

    python -m eval.bench                         # all tasks with data, default models
    python -m eval.bench --tasks sentiment summarization --limit 10
    python -m eval.bench --models summarization=facebook/bart-large-cnn

Results are written to outputs/bench/bench_<timestamp>.{json,md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from core.config import ROOT, AppConfig, load_config
from core.device import set_seed
from eval import metrics
from models import tasks as T
from models.registry import ModelRegistry

DATA = ROOT / "data"
BENCH_TASKS = ["sentiment", "summarization", "next_word", "image_gen"]


def _read_csv(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))[:limit]


def bench_sentiment(reg: ModelRegistry, limit: int, model: str | None) -> dict[str, Any]:
    rows = _read_csv(DATA / "sentiment" / "sst2_validation.csv", limit)
    texts, gold = [r["text"] for r in rows], [r["label"] for r in rows]
    reg.get("sentiment", model)  # load outside the timed region
    t0 = time.perf_counter()
    pred = T.classify_batch(reg, texts, model)
    dt = time.perf_counter() - t0
    rep = metrics.classification_report(gold, pred)
    return {"task": "sentiment", "model": model or reg.cfg.tasks["sentiment"].model, "n": len(rows),
            "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"],
            "confusion": f"{rep['labels']} {rep['confusion_matrix']}",
            "latency_ms/sample": round(1000 * dt / len(rows), 1)}


def bench_summarization(reg: ModelRegistry, limit: int, model: str | None) -> dict[str, Any]:
    rows = _read_csv(DATA / "summarization" / "cnn_dailymail_test.csv", limit)
    p = reg.cfg.tasks["summarization"].params
    preds, lat = [], []
    for r in rows:
        res = T.summarize(reg, r["source"], model, p.get("max_new_tokens", 130), p.get("min_new_tokens", 30),
                          p.get("num_beams", 4))
        preds.append(res.output)
        lat.append(res.meta["latency_s"])
    rg = metrics.rouge(preds, [r["summary"] for r in rows])
    return {"task": "summarization", "model": model or reg.cfg.tasks["summarization"].model, "n": len(rows),
            "rouge1": rg["rouge1"], "rouge2": rg["rouge2"], "rougeL": rg["rougeL"],
            "len_ratio": metrics.length_ratio(preds, [r["source"] for r in rows]),
            "latency_s (median)": round(statistics.median(lat), 3)}


def bench_next_word(reg: ModelRegistry, limit: int, model: str | None) -> dict[str, Any]:
    text = (DATA / "next_word" / "wikitext2_test.txt").read_text(encoding="utf-8")
    paragraphs = [p for p in text.split("\n") if len(p.split()) > 20][: max(limit, 5)]
    tok, lm = reg.get("next_word", model)
    t0 = time.perf_counter()
    ppl = metrics.perplexity(lm, tok, paragraphs, device=reg.device)
    dt = time.perf_counter() - t0
    # streaming latency: time to first token and throughput for a fixed prompt
    meta: dict[str, Any] = {}
    for _, meta in T.stream_continuation(reg, "The history of the city begins", model, max_new_tokens=20,
                                         temperature=0):
        pass
    return {"task": "next_word", "model": model or reg.cfg.tasks["next_word"].model,
            "n": len(paragraphs), "perplexity": ppl, "eval_s": round(dt, 2),
            "first_token_ms": round(1000 * meta.get("first_token_s", 0), 1),
            "tokens/s": round(meta.get("tokens_per_s", 0), 1)}


def bench_image_gen(reg: ModelRegistry, limit: int, model: str | None, out_dir: Path) -> dict[str, Any]:
    rows = _read_csv(ROOT / "assets" / "prompts.csv", limit)
    p = reg.cfg.tasks["image_gen"].params
    images, prompts, lat = [], [], []
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        prompt = f"{r['prompt']}, {r['style']}"
        res = T.generate_image(reg, prompt, model, steps=p.get("steps", 2),
                               guidance_scale=p.get("guidance_scale", 0.0),
                               width=p.get("width", 512), height=p.get("height", 512), seed=i)
        res.output.save(img_dir / f"{i:02d}.png")
        images.append(res.output)
        prompts.append(prompt)
        lat.append(res.meta["latency_s"])
    clip = metrics.ClipScorer(reg.cfg.eval.get("clip_model", "openai/clip-vit-base-patch32"), "cpu")
    scores = clip.score(images, prompts)
    return {"task": "image_gen", "model": model or reg.cfg.tasks["image_gen"].model, "n": len(rows),
            "clip_score": round(statistics.mean(scores), 2), "steps": p.get("steps", 2),
            "s/image (median)": round(statistics.median(lat), 2)}


def run_benchmark(cfg: AppConfig, task_names: list[str], limit: int,
                  models: dict[str, str] | None = None, reg: ModelRegistry | None = None,
                  progress: Callable[[str], None] | None = None) -> tuple[list[dict[str, Any]], Path]:
    """Run the benchmark and save JSON + markdown; returns (rows, markdown_path)."""
    set_seed(int(cfg.app.get("seed", 42)))
    reg = reg or ModelRegistry(cfg)
    models = models or {}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.outputs_dir / "bench" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name in task_names:
        if progress:
            progress(name)
        fn: Callable[..., dict[str, Any]] = {
            "sentiment": bench_sentiment, "summarization": bench_summarization,
            "next_word": bench_next_word,
        }.get(name) or (lambda r, l, m: bench_image_gen(r, l, m, out_dir))
        try:
            row = fn(reg, limit, models.get(name))
        except Exception as exc:  # keep going so one failure doesn't sink the run
            row = {"task": name, "error": f"{type(exc).__name__}: {exc}"}
        row["device"] = reg.device
        rows.append(row)
        print(json.dumps(row, default=str))
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    md_path = out_dir / "results.md"
    body = "\n\n".join(f"### {r['task']}\n\n" + metrics.markdown_table([r]) for r in rows)
    md_path.write_text(f"# Benchmark {stamp}\n\n{body}\n")
    return rows, md_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tasks", nargs="+", default=BENCH_TASKS, choices=BENCH_TASKS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=[], help="task=model_id overrides")
    args = ap.parse_args()
    cfg = load_config(args.config)
    limit = args.limit or int(cfg.eval.get("bench_limit", 20))
    overrides = dict(m.split("=", 1) for m in args.models)
    rows, md = run_benchmark(cfg, args.tasks, limit, overrides)
    print()
    print(md.read_text())


if __name__ == "__main__":
    main()
