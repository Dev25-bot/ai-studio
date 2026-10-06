"""Fast unit tests for the chat assistant's routing and text handling (no model loads)."""
from __future__ import annotations

import pytest

from core.safety import UserFacingError
from eval.llm_eval import final_answer, grade
from models.assistant import Turn, image_request, read_document, tidy_reply


@pytest.mark.parametrize("text, prompt", [
    ("draw a cat wearing a hat", "a cat wearing a hat"),
    ("/imagine neon city at night", "neon city at night"),
    ("Generate an image of a red fox in snow", "a red fox in snow"),
    ("can you create a picture of mountains at sunrise", "mountains at sunrise"),
    ("make me a logo for a coffee shop", "logo for a coffee shop"),
    ("make a cartoon portrait of a dog", "cartoon portrait of a dog"),
])
def test_image_request_detected(text, prompt):
    assert image_request(text) == prompt


@pytest.mark.parametrize("text", [
    "generate a python function to sort a list", "create a summary of this", "draw a conclusion from the data",
    "what is an image classifier?", "Write me a poem", "/image", "",
])
def test_image_request_not_triggered(text):
    assert image_request(text) is None


@pytest.mark.parametrize("raw, clean", [
    ("Answer: Paris", "Paris"),
    ("Count...\n\nFinal answer: **3**\n\nFinal answer: 3", "Count...\n\n**Answer: 3**"),
    ("x = 3\n\nFinal Answer:\n\nx = 2 or x = 3\n\nAnswer: 2 and 3", "x = 3\n\nx = 2 or x = 3\n\n**Answer: 2 and 3**"),
    ("Rain on glass.\nNature song.**", "Rain on glass.\nNature song."),
    ("Use **bold** here", "Use **bold** here"),
    ("The answer is clear: yes.", "The answer is clear: yes."),
])
def test_tidy_reply(raw, clean):
    assert tidy_reply(raw) == clean


def test_read_text_document_and_truncation(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Title\n\n\n\nBody " + "x" * 100)
    text, truncated = read_document(p, max_chars=1000)
    assert text.startswith("# Title\n\nBody") and not truncated
    text, truncated = read_document(p, max_chars=20)
    assert len(text) == 20 and truncated


def test_read_docx(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Revenue grew 12 percent.")
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "India", "2.5M"
    p = tmp_path / "r.docx"
    d.save(p)
    text, _ = read_document(p, 1000)
    assert "Revenue grew 12 percent." in text and "India | 2.5M" in text


def test_read_document_rejects_bad_files(tmp_path):
    (tmp_path / "a.zip").write_bytes(b"PK")
    with pytest.raises(UserFacingError, match="unsupported"):
        read_document(tmp_path / "a.zip", 100)
    (tmp_path / "empty.txt").write_text("   ")
    with pytest.raises(UserFacingError, match="empty"):
        read_document(tmp_path / "empty.txt", 100)


def test_turn_roundtrip_and_model_content():
    t = Turn("user", "Summarise", docs=[{"name": "a.txt", "text": "hello", "truncated": True}])
    assert Turn.from_dict(t.to_dict()) == t
    assert '<document name="a.txt" (truncated)>\nhello\n</document>' in t.model_content()


def test_eval_grader():
    assert final_answer("work...\nFinal answer: 42") == "42"
    assert grade({"re": r"\b42\b"}, "so...\n**Answer: 42**")
    assert not grade({"re": r"\b42\b"}, "Answer: 41")
    assert grade({"re_all": ["chat", "dort"]}, "Le chat dort.")
    assert grade({"test": "assert f(2) == 4"}, "```python\ndef f(x):\n    return x * 2\n```")
    assert not grade({"test": "assert f(2) == 5"}, "```python\ndef f(x):\n    return x * 2\n```")
