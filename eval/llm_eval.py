"""Hard-question evaluation for the chat model: accuracy + speed, per model.

    python -m eval.llm_eval --models mlx-community/Qwen3-4B-Instruct-2507-4bit
    python -m eval.llm_eval --models a b c --vision b      # 'b' is a vision-language checkpoint

Each question in eval/hard_questions.jsonl is graded automatically:
  re        regex that must match the final answer (case-insensitive unless "case": true)
  re_all    every regex must match
  test      Python asserts run against the code the model wrote (subprocess, 10 s timeout)
Results go to outputs/llm_eval/<timestamp>/results.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from core.config import ROOT
from models.llm import ChatEngine, EngineConfig

QUESTIONS = ROOT / "eval" / "hard_questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def final_answer(text: str) -> str:
    """The text after the last 'Answer:' marker, or the whole reply if there is none."""
    m = list(re.finditer(r"answer\s*[:：]\s*\**", text, re.IGNORECASE))
    return text[m[-1].end():].strip() if m else text


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len) if blocks else text


def grade(q: dict[str, Any], reply: str) -> bool:
    if "test" in q:
        code = extract_code(reply) + "\n\n" + q["test"] + "\nprint('OK')\n"
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0 and "OK" in r.stdout
    target = final_answer(reply)
    flags = 0 if q.get("case") else re.IGNORECASE
    if "re" in q:
        return re.search(q["re"], target, flags) is not None
    return all(re.search(p, reply, flags) for p in q["re_all"])


def evaluate(model: str, vision: bool, questions: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    eng = ChatEngine(EngineConfig(model=model, vision=vision, temperature=0.0, max_tokens=max_tokens))
    t_load = time.perf_counter()
    eng.load()
    load_s = time.perf_counter() - t_load
    rows = []
    for q in questions:
        msgs = eng.build_messages([{"role": "user", "content": q["q"]}])
        reply, last = eng.complete(msgs)
        ok = grade(q, reply)
        rows.append({"id": q["id"], "cat": q["cat"], "ok": ok, "first_token_s": last.first_token_s,
                     "tokens_per_s": last.tokens_per_s, "tokens": last.generated_tokens,
                     "elapsed_s": last.elapsed_s, "reply": reply})
        print(f"  [{'✓' if ok else '✗'}] {q['id']:12s} {last.elapsed_s:5.1f}s {last.tokens_per_s:5.1f} tok/s",
              flush=True)
    eng.unload()
    cats: dict[str, list[bool]] = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r["ok"])
    return {
        "model": model, "vision": vision, "load_s": round(load_s, 1),
        "accuracy": sum(r["ok"] for r in rows) / len(rows),
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in cats.items()},
        "median_first_token_s": round(median(r["first_token_s"] or 0 for r in rows), 2),
        "mean_tokens_per_s": round(mean(r["tokens_per_s"] for r in rows), 1),
        "median_reply_s": round(median(r["elapsed_s"] for r in rows), 1),
        "failed": [r["id"] for r in rows if not r["ok"]],
        "rows": rows,
    }


def to_markdown(results: list[dict[str, Any]]) -> str:
    cats = sorted({c for r in results for c in r["by_category"]})
    head = "| model | accuracy | " + " | ".join(cats) + " | first token (s) | tok/s | median reply (s) | load (s) |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in sorted(results, key=lambda r: -r["accuracy"]):
        lines.append(f"| `{r['model']}` | **{r['accuracy']:.0%}** | "
                     + " | ".join(r["by_category"].get(c, "-") for c in cats)
                     + f" | {r['median_first_token_s']} | {r['mean_tokens_per_s']} | {r['median_reply_s']}"
                     f" | {r['load_s']} |")
    lines.append("")
    for r in results:
        lines.append(f"- `{r['model']}` failed: {', '.join(r['failed']) or 'none'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--vision", nargs="*", default=[], help="models that are vision-language checkpoints")
    ap.add_argument("--only", nargs="*", default=[], help="question ids to run")
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()
    qs = [q for q in load_questions() if not args.only or q["id"] in args.only]
    out = ROOT / "outputs" / "llm_eval" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for m in args.models:
        print(f"\n=== {m}", flush=True)
        results.append(evaluate(m, m in args.vision, qs, args.max_tokens))
        (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    md = to_markdown(results)
    (out / "results.md").write_text(md)
    print("\n" + md + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
