# Implementation Plan — Multi-Task Gradio AI Studio (PROJECT 1211)

> **Update (2026-10-07):** the eight-tab UI described below was later replaced by a single ChatGPT-style chat backed by a local MLX model (Qwen3-VL-4B). See README.md for the current design; the task modules below still power `python -m eval.bench`.

> **Status: all 7 phases complete (2026-10-07).** The eight tabs were checked with real models on M2/MPS, and 10 unit tests plus 3 integration tests pass.
> Benchmark results are in README.md. Not done: a screencast/GIF, and building the Docker image (Docker isn't installed on this machine).

Source PRD: `AI_ML_Multi_task_web_app.pdf` (GUVI × HCL).

## Target hardware (detected)
Apple M2 · 8 GB unified memory · macOS · no CUDA → **MPS** (Apple GPU) with CPU fallback.
Consequences: lazy model loading, LRU cache capped by `max_loaded_models`, fp16 only for diffusion on MPS,
CPU-friendly default checkpoints (sd-turbo instead of SDXL, `gpt2` instead of `gpt2-medium`).

## Architecture
```
app.py                     Gradio Blocks UI (tabs, examples, downloads, footer metrics, dark mode, shortcuts)
configs/app.yaml           Toggles tabs, picks checkpoints/backends, device, limits, logging
core/config.py             YAML loader → typed dataclasses
core/device.py             CUDA/MPS/CPU detection, dtype, autocast context, seeding
core/safety.py             Input-length limits, blocklist prompt filter, token-bucket rate limiter
core/logging_utils.py      Structured JSON-lines logging (+ optional W&B)
core/io_utils.py           Save outputs (PNG/TXT/JSON) to outputs/
models/registry.py         Factory + thread-safe LRU cache; backend switch: hf | local | remote
models/tasks.py            Task runners: image_gen, summarize, sentiment, next_word (streaming),
                           translate, ner, qa, image_classify
models/remote.py           OpenAI-compatible remote endpoint client (flag-switchable)
eval/metrics.py            ROUGE-1/2/L, accuracy/macro-F1/confusion matrix, CLIPScore, perplexity
eval/bench.py              Benchmark mode CLI → metrics table (markdown + JSON in outputs/bench/)
data/<task>/               Small permissively licensed eval subsets (CSV/JSON/TXT)
assets/prompts.csv         Curated image prompts (prompt,style)
examples/                  Demo inputs
tests/                     pytest: config, safety, metrics, registry (no model downloads)
README.md, MODEL_CARDS.md, Makefile, requirements.txt, environment.yml, Dockerfile
```

## Default models (swap in YAML)
| Tab | Default | Alt |
|---|---|---|
| Image Gen | `stabilityai/sd-turbo` (1–4 steps, MPS-friendly) | `runwayml/stable-diffusion-v1-5`, SDXL |
| Summarization | `sshleifer/distilbart-cnn-12-6` | `facebook/bart-large-cnn`, `google/flan-t5-base` |
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | — |
| Next-word | `gpt2` | `gpt2-medium` |
| Translation | `Helsinki-NLP/opus-mt-en-de` | other opus-mt pairs |
| NER | `dslim/bert-base-NER` | — |
| Q&A | `deepset/roberta-base-squad2` | — |
| Image classify | `google/vit-base-patch16-224` | — |
| CLIPScore | `openai/clip-vit-base-patch32` | — |

## Phases
1. **Env & skeleton** — Python 3.11 venv via `uv`, deps, config + registry + device + safety + logging.
2. **Core tabs** — Sentiment, Summarization (fast), then Next-Word (streaming), Image Gen.
3. **Optional tabs** — Translation, NER, Q&A, Image Classification.
4. **UX polish** — examples per tab, download buttons, latency/token/step footer, dark mode, keyboard shortcuts, tooltips, disclaimers.
5. **Eval** — metrics adapters + `bench.py`; build small datasets; run benchmark; write comparative notes.
6. **Repro & deploy** — tests, Makefile, requirements/environment.yml, Dockerfile, README with screenshot.
7. **Verify** — pytest, launch app, drive each tab in the browser pane, screenshot into `docs/`.

## Acceptance checks
- `make test` green; `make run` serves on :7860 and every enabled tab returns output.
- `make bench` prints a metrics table (ROUGE, Acc/F1, perplexity, CLIPScore, latency).
- Disabling a tab in `configs/app.yaml` removes it from the UI.
