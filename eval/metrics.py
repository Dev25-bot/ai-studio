"""Metric adapters: ROUGE, accuracy / macro-F1 / confusion matrix, CLIPScore, perplexity."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


def rouge(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Mean ROUGE-1/2/L F-measures (×100)."""
    from rouge_score import rouge_scorer

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(predictions, references):
        s = scorer.score(r, p)
        for k in totals:
            totals[k] += s[k].fmeasure
    n = max(1, len(predictions))
    return {k: round(100 * v / n, 2) for k, v in totals.items()}


def length_ratio(predictions: Sequence[str], sources: Sequence[str]) -> float:
    """Average summary length / source length (in words)."""
    ratios = [len(p.split()) / max(1, len(s.split())) for p, s in zip(predictions, sources)]
    return round(float(np.mean(ratios)) if ratios else 0.0, 4)


def classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro-F1 and a labelled confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels)), 4),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


class ClipScorer:
    """CLIPScore = 100 · max(cos(text_emb, image_emb), 0)  (Hessel et al., 2021)."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def score(self, images: Sequence[Image.Image], prompts: Sequence[str]) -> list[float]:
        inputs = self.processor(text=list(prompts), images=list(images), return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        img = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        txt = self.model.get_text_features(input_ids=inputs["input_ids"],
                                           attention_mask=inputs["attention_mask"])
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        cos = (img * txt).sum(-1).clamp(min=0)
        return [round(100 * float(c), 2) for c in cos]


@torch.inference_mode()
def perplexity(model: Any, tokenizer: Any, texts: Sequence[str], device: str = "cpu",
               max_length: int = 512, stride: int = 256) -> float:
    """Sliding-window perplexity of a causal LM over the concatenated texts."""
    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    ids = enc.input_ids
    seq_len = ids.size(1)
    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = ids[:, begin:end].to(device)
        target = input_ids.clone()
        target[:, :-trg_len] = -100
        loss = model(input_ids, labels=target).loss
        valid = int((target[:, 1:] != -100).sum())
        nll_sum += float(loss) * valid
        n_tokens += valid
        prev_end = end
        if end == seq_len:
            break
    return round(math.exp(nll_sum / max(1, n_tokens)), 3)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render a list of dicts as a GitHub-flavoured markdown table."""
    if not rows:
        return "_no results_"
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)
