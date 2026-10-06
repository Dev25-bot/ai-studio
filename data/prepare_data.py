"""Download small, permissively licensed evaluation subsets via the HF datasets-server API.

    python data/prepare_data.py --n 100

Writes one directory per task:
  data/summarization/cnn_dailymail_test.csv   (id, split, source, summary)   — Apache-2.0
  data/sentiment/sst2_validation.csv          (id, split, text, label)       — SST-2 / GLUE
  data/next_word/wikitext2_test.txt           plain text                    — CC BY-SA 3.0
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
DATA = Path(__file__).resolve().parent


def fetch_rows(dataset: str, config: str, split: str, n: int, offset: int = 0) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < n:
        length = min(100, n - len(rows))
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": offset + len(rows), "length": length})
        with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
            batch = [x["row"] for x in json.loads(r.read())["rows"]]
        if not batch:
            break
        rows.extend(batch)
    return rows


def main(n: int) -> None:
    (DATA / "summarization").mkdir(exist_ok=True)
    rows = fetch_rows("abisee/cnn_dailymail", "3.0.0", "test", n)
    with (DATA / "summarization" / "cnn_dailymail_test.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "source", "summary"])
        for r in rows:
            w.writerow([r["id"], "test", r["article"], r["highlights"].replace("\n", " ")])
    print(f"summarization: {len(rows)} rows")

    (DATA / "sentiment").mkdir(exist_ok=True)
    rows = fetch_rows("nyu-mll/glue", "sst2", "validation", max(n, 200))
    with (DATA / "sentiment" / "sst2_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "text", "label"])
        for r in rows:
            w.writerow([r["idx"], "validation", r["sentence"].strip(), "POSITIVE" if r["label"] == 1 else "NEGATIVE"])
    print(f"sentiment: {len(rows)} rows")

    (DATA / "next_word").mkdir(exist_ok=True)
    rows = fetch_rows("Salesforce/wikitext", "wikitext-2-raw-v1", "test", 400)
    text = "".join(r["text"] for r in rows).strip()
    (DATA / "next_word" / "wikitext2_test.txt").write_text(text, encoding="utf-8")
    print(f"next_word: {len(text.split())} words")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    main(ap.parse_args().n)
