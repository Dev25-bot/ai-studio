"""Integration tests that load small models. Run with:  pytest -m integration"""
from __future__ import annotations

import pytest

from core.config import load_config
from models import tasks as T
from models.registry import ModelRegistry

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def reg() -> ModelRegistry:
    return ModelRegistry(load_config())


def test_sentiment(reg):
    assert T.classify_sentiment(reg, "I absolutely love this!").output["label"] == "POSITIVE"
    assert T.classify_sentiment(reg, "This is terrible and broken.").output["label"] == "NEGATIVE"


def test_next_word_stream(reg):
    chunks = list(T.stream_continuation(reg, "The capital of France is", max_new_tokens=5, temperature=0))
    assert chunks and chunks[-1][1]["new_tokens"] >= 1
    assert len(T.top_next_tokens(reg, "Hello", k=3)) == 3


def test_summarize(reg):
    text = "The council approved a new transit plan. " * 30
    res = T.summarize(reg, text, max_new_tokens=40, min_new_tokens=5, num_beams=1)
    assert 0 < len(res.output.split()) < len(text.split())
