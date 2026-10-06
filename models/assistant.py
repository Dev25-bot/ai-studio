"""The one-stop assistant behind the chat UI.

Routes every message to the right capability:
- "draw / generate an image of …"  → text-to-image (sd-turbo), shown inline in the chat
- attached images                  → vision-language model (describe, read text, answer about it)
- attached documents (pdf/docx/txt/code/csv…) → inlined as context for summaries, translation, Q&A
- everything else                  → chat model (questions, maths, code, writing, analysis)

Only one heavy model is kept in memory at a time so the app fits in 8 GB of RAM.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageOps

from core.config import AppConfig
from core.io_utils import save_image
from core.safety import UserFacingError, check_image_prompt
from models.llm import ChatEngine, EngineConfig

log = logging.getLogger("ai_studio.assistant")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic"}
TEXT_EXTS = {".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm", ".log", ".ini",
             ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".rb",
             ".php", ".sql", ".sh", ".swift", ".kt", ".r", ".tex", ".css", ".toml"}
DOC_EXTS = {".pdf", ".docx"} | TEXT_EXTS

_DRAW = re.compile(
    r"^\s*(?:/(?:image|img|draw|imagine)\b\s*(?P<a>.*)"
    r"|(?:(?:please|pls|can you|could you|would you|kindly)[\s,]+)*"
    r"(?:(?:draw|paint|sketch|illustrate)\b(?!\s+(?:a\s+)?(?:conclusion|comparison|the line|parallels?|attention))"
    r"|(?:generate|create|make|render|design|produce|show)\s+(?:me\s+)?(?:an?\s+|the\s+|some\s+)?"
    r"(?=(?:\w+\s+)?(?:image|picture|photo|illustration|drawing|painting|artwork|art|wallpaper|logo|poster|"
    r"portrait|render)s?\b))\s*(?P<b>.*))$",
    re.IGNORECASE | re.DOTALL,
)
_DRAW_NOUN = re.compile(r"^(?:me\s+)?(?:an?\s+|the\s+|some\s+)?(?:image|picture|photo|illustration|drawing|"
                        r"painting|artwork|art|render)s?\s+(?:of|showing|that shows)\s+", re.IGNORECASE)


_FINAL = re.compile(r"^[ \t>*_-]*(?:final\s+)?answer\s*[:：][ \t*_]*(.+?)[ \t*_.]*$", re.IGNORECASE | re.MULTILINE)


def tidy_reply(text: str) -> str:
    """Keep exactly one bold 'Answer:' line at the end and drop stray markdown left by the model."""
    # A bare "Final answer:" heading (answer on the following lines) is redundant once we add the bold line.
    text = re.sub(r"^[ \t>#*_-]*(?:final\s+)?answer\s*[:：]?[ \t*_]*$\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    finals = _FINAL.findall(text)
    if finals:
        body = _FINAL.sub("", text).rstrip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        answer = finals[-1].strip().strip("*").strip()
        text = f"{body}\n\n**Answer: {answer}**" if body else answer
    if text.count("**") % 2:  # unmatched bold marker, usually a dangling "**" at the end
        text = text[::-1].replace("**", "", 1)[::-1]
    return text.strip()


def image_request(text: str) -> str | None:
    """Return the image prompt if the message asks for a picture, else None."""
    m = _DRAW.match(text or "")
    if not m:
        return None
    prompt = (m.group("a") if m.group("a") is not None else m.group("b") or "").strip()
    prompt = _DRAW_NOUN.sub("", prompt, count=1).strip(" .:!")
    return prompt or None


def read_document(path: str | Path, max_chars: int) -> tuple[str, bool]:
    """Extract text from a supported document. Returns (text, truncated)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(p))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise UserFacingError(f"Could not read the PDF '{p.name}': {exc}") from exc
        if not text.strip():
            raise UserFacingError(f"'{p.name}' has no selectable text (it may be a scanned PDF). "
                                  "Try uploading a screenshot of the page as an image instead.")
    elif ext == ".docx":
        import docx

        d = docx.Document(str(p))
        parts = [para.text for para in d.paragraphs]
        for table in d.tables:
            parts += [" | ".join(c.text for c in row.cells) for row in table.rows]
        text = "\n".join(parts)
    elif ext in TEXT_EXTS:
        text = p.read_bytes().decode("utf-8", errors="replace")
    else:
        raise UserFacingError(f"'{p.name}': unsupported file type '{ext or '?'}'. "
                              "Attach images, PDFs, Word (.docx) or text/code files.")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise UserFacingError(f"'{p.name}' is empty.")
    return (text[:max_chars], True) if len(text) > max_chars else (text, False)


def prepare_image(path: str | Path, max_side: int, out_dir: Path) -> str:
    """Fix EXIF rotation, convert to RGB and downscale so vision prefill stays fast."""
    p = Path(path)
    try:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    except Exception as exc:
        raise UserFacingError(f"Could not open the image '{p.name}'.") from exc
    img.thumbnail((max_side, max_side))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{p.stem[:40]}_{int(time.time() * 1000)}.jpg"
    img.save(dst, quality=92)
    return str(dst)


@dataclass
class Turn:
    role: str                                    # "user" | "assistant"
    text: str                                    # what the user typed / the model answered
    images: list[str] = field(default_factory=list)
    docs: list[dict[str, Any]] = field(default_factory=list)   # {"name", "text", "truncated"}
    image_out: str | None = None                 # generated picture (assistant turns)

    def model_content(self) -> str:
        """Content as the language model sees it (documents inlined)."""
        parts = [self.text] if self.text else []
        for d in self.docs:
            note = " (truncated)" if d["truncated"] else ""
            parts.append(f'<document name="{d["name"]}"{note}>\n{d["text"]}\n</document>')
        if self.image_out:
            parts.append("[I generated the requested image and showed it to the user.]")
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "images": self.images, "docs": self.docs,
                "image_out": self.image_out}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        return cls(d["role"], d.get("text", ""), list(d.get("images") or []), list(d.get("docs") or []),
                   d.get("image_out"))


@dataclass
class Reply:
    text: str = ""
    image: str | None = None
    status: str = ""          # short progress note shown while working ("Reading document…")
    stats: dict[str, Any] = field(default_factory=dict)
    done: bool = False


class Assistant:
    def __init__(self, cfg: AppConfig, registry: Any) -> None:
        c = cfg.chat
        self.cfg = cfg
        self.reg = registry
        self.engine = ChatEngine(EngineConfig(
            model=c.get("model", "mlx-community/Qwen3-VL-4B-Instruct-4bit"), vision=bool(c.get("vision", True)),
            max_tokens=int(c.get("max_tokens", 1536)), temperature=float(c.get("temperature", 0.3)),
            context_tokens=int(c.get("context_tokens", 6144)), keepalive_s=float(c.get("keepalive_s", 15))))
        self.max_doc_chars = int(c.get("max_document_chars", 16000))
        self.max_image_side = int(c.get("max_image_side", 1024))
        self.uploads = cfg.outputs_dir / "uploads"
        self.out = cfg.outputs_dir
        self._heavy = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.engine.cfg.model.split("/")[-1]

    # ------------------------------------------------------------------ inputs
    def make_user_turn(self, text: str, files: list[str]) -> Turn:
        images, docs = [], []
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in IMAGE_EXTS:
                if not self.engine.cfg.vision:
                    raise UserFacingError("The current model can't see images. Switch to a vision model in "
                                          "configs/app.yaml (chat.vision: true).")
                images.append(prepare_image(f, self.max_image_side, self.uploads))
            else:
                body, truncated = read_document(f, self.max_doc_chars)
                docs.append({"name": Path(f).name, "text": body, "truncated": truncated})
        text = (text or "").strip()
        if not text:
            text = ("Describe this image in detail." if images and not docs
                    else "Summarize this document: key points, then a short conclusion." if docs else "")
        if not text:
            raise UserFacingError("Type a message or attach a file first.")
        return Turn("user", text, images, docs)

    # ------------------------------------------------------------------ memory
    def _use_llm(self) -> None:
        """Free the diffusion model before the chat model runs (8 GB budget)."""
        if "image_gen" in {k.split(":")[0] for k in self.reg.loaded()}:
            self.reg.unload_task("image_gen")

    def _use_diffusion(self) -> None:
        if self.engine.loaded:
            self.engine.unload()

    def warmup(self) -> None:
        self.engine.load(warm=True)
        self.engine.start_keepalive()

    # ------------------------------------------------------------------ respond
    def respond(self, history: list[Turn], user: Turn, temperature: float | None = None,
                max_tokens: int | None = None, system_prompt: str | None = None) -> Iterator[Reply]:
        prompt = image_request(user.text) if not user.images and not user.docs else None
        with self._heavy:
            if prompt:
                yield from self._draw(prompt)
            else:
                yield from self._chat(history, user, temperature, max_tokens, system_prompt)

    def _draw(self, prompt: str) -> Iterator[Reply]:
        check_image_prompt(prompt)
        yield Reply(status=f"🎨 Painting “{prompt}”…")
        self._use_diffusion()
        from models import tasks as T

        p = self.cfg.tasks["image_gen"].params if "image_gen" in self.cfg.tasks else {}
        res = T.generate_image(self.reg, prompt, None, "", int(p.get("steps", 2)),
                               float(p.get("guidance_scale", 0.0)), int(p.get("width", 512)),
                               int(p.get("height", 512)), int(time.time()) % 100000)
        path = save_image(res.output, self.out, "image_gen", prompt)
        yield Reply(text=f"Here's your image of **{prompt}**.", image=path, done=True,
                    stats={"latency_s": res.meta["latency_s"], "kind": "image", "model": "sd-turbo"})

    def _chat(self, history: list[Turn], user: Turn, temperature: float | None, max_tokens: int | None,
              system_prompt: str | None) -> Iterator[Reply]:
        if not self.engine.loaded:
            yield Reply(status="⏳ Loading the model (first message only)…")
        self._use_llm()
        self.engine.load()
        if user.docs:
            yield Reply(status=f"📄 Reading {', '.join(d['name'] for d in user.docs)}…")
        elif user.images:
            yield Reply(status="🖼️ Looking at the image…")

        msgs = [{"role": t.role, "content": t.model_content()} for t in history + [user]]
        msgs = self.engine.build_messages(msgs, system_prompt)
        # Vision: the image(s) from this message, or the latest earlier image for follow-ups ("what colour is it?").
        images = user.images or next((t.images for t in reversed(history) if t.images), [])
        text, last = "", None
        for chunk in self.engine.stream(msgs, images=images, max_tokens=max_tokens, temperature=temperature):
            text += chunk.text
            last = chunk
            yield Reply(text=text)
        stats = {"kind": "chat", "model": self.model_name}
        if last is not None:
            stats.update(first_token_s=last.first_token_s, tokens_per_s=last.tokens_per_s,
                         tokens=last.generated_tokens, prompt_tokens=last.prompt_tokens,
                         latency_s=last.elapsed_s, finish=last.finish_reason)
        text = tidy_reply(text)
        if not text.strip():
            text = "_(No answer was generated — please try rephrasing.)_"
        elif last is not None and last.finish_reason == "length":
            text += "\n\n_…reply cut off at the length limit. Say **continue** for more._"
        yield Reply(text=text, stats=stats, done=True)
