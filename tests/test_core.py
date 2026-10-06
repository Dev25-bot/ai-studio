"""Fast unit tests — no model downloads."""
from __future__ import annotations

import textwrap

import pytest

from core.config import ConfigError, load_config
from core.device import resolve_device
from core.safety import RateLimiter, UserFacingError, check_image_prompt, check_length, clean_text
from eval import metrics
from models.registry import ModelRegistry


def test_default_config_loads():
    cfg = load_config()
    names = [t.name for t in cfg.enabled_tasks]
    assert names[:4] == ["image_gen", "summarization", "sentiment", "next_word"]
    assert cfg.tasks["sentiment"].model_choices[0] == cfg.tasks["sentiment"].model


def test_config_toggle_and_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        tasks:
          sentiment: {enabled: true, model: some/model}
          ner: {enabled: false, model: x}
    """))
    cfg = load_config(p)
    assert [t.name for t in cfg.enabled_tasks] == ["sentiment"]

    p.write_text("tasks:\n  bogus: {model: x}\n")
    with pytest.raises(ConfigError):
        load_config(p)
    p.write_text("tasks:\n  sentiment: {backend: ftp, model: x}\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_resolve_device_fallback():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_clean_and_length():
    assert clean_text("  “hello”   world \n") == '"hello" world'
    with pytest.raises(UserFacingError):
        check_length("   ", 10)
    with pytest.raises(UserFacingError):
        check_length("x" * 11, 10)
    assert check_length("ok", 10) == "ok"


def test_image_prompt_filter():
    check_image_prompt("a lighthouse at sunset")
    with pytest.raises(UserFacingError):
        check_image_prompt("NSFW picture")


def test_rate_limiter():
    rl = RateLimiter(2)
    rl.check("a")
    rl.check("a")
    with pytest.raises(UserFacingError):
        rl.check("a")
    rl.check("b")  # independent per key
    RateLimiter(0).check("a")  # disabled


def test_rouge_and_classification():
    r = metrics.rouge(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert r["rouge1"] == 100.0 and r["rougeL"] == 100.0
    rep = metrics.classification_report(["POS", "NEG", "POS"], ["POS", "POS", "POS"])
    assert rep["accuracy"] == pytest.approx(0.6667, abs=1e-3)
    assert rep["confusion_matrix"] == [[0, 1], [0, 2]]
    assert metrics.length_ratio(["a b"], ["a b c d"]) == 0.5


def test_markdown_table():
    md = metrics.markdown_table([{"a": 1, "b": 2}])
    assert md.splitlines()[0] == "| a | b |"


def test_registry_errors():
    cfg = load_config()
    reg = ModelRegistry(cfg)
    with pytest.raises(UserFacingError):
        reg.get("not_a_task")
    cfg.tasks["ner"].backend = "remote"
    with pytest.raises(UserFacingError):
        reg.get("ner")  # NER has no remote implementation
    cfg.tasks["qa"].backend = "local"
    with pytest.raises(UserFacingError):
        reg.get("qa", "/nonexistent/checkpoint")


def test_remote_requires_key(monkeypatch):
    from models.remote import RemoteClient

    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(UserFacingError):
        RemoteClient("http://localhost:1", "NO_SUCH_KEY", "m").chat("s", "u")
