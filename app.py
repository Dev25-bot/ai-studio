"""AI Studio — one private chat for everything, running locally on Apple Silicon.

    python app.py                    # uses configs/app.yaml
    python app.py --config my.yaml --port 7861 --share

Ask questions, write code, attach PDFs/Word/text/code files to summarise or translate them, attach photos
to ask about them, or say "draw …" to generate an image — all in a single conversation.
"""
from __future__ import annotations

import argparse
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from core.config import ROOT, AppConfig, load_config
from core.device import device_summary, set_seed
from core.logging_utils import EventLogger
from core.safety import RateLimiter, UserFacingError, check_length
from models.assistant import DOC_EXTS, IMAGE_EXTS, Assistant, Reply, Turn
from models.registry import ModelRegistry

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"gradio(\.|$)")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*Gradio 6\.0.*")

MAX_SAVED_CHATS = 30
IDLE_BOX = dict(value=None, submit_btn=True, stop_btn=False)  # composer state between answers
FILE_TYPES = sorted(IMAGE_EXTS | DOC_EXTS)

SUGGESTIONS = [
    {"text": "Explain how transformers work, like I'm a CS student", "display_text": "🧠 Explain transformers"},
    {"text": "Write a Python function that checks if a string is a valid IPv4 address, with tests",
     "display_text": "💻 Write & test code"},
    {"text": "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
     "display_text": "🧩 Solve a trick question"},
    {"text": "Draw a cozy cabin in a snowy forest at night, warm light in the windows",
     "display_text": "🎨 Draw a picture"},
    {"text": "Plan a 3-day trip to Jaipur on a student budget", "display_text": "🗺️ Plan a trip"},
    {"text": "Translate to French and Spanish: 'Where is the nearest train station?'",
     "display_text": "🌍 Translate"},
]

CSS = """
:root {--brand-grad: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #d946ef 100%);}
.gradio-container {max-width: 100% !important; padding: 0 !important;}
footer {display: none !important;}
body {background-image: radial-gradient(900px 420px at 15% -10%, rgba(99,102,241,.14), transparent 60%),
  radial-gradient(800px 380px at 100% 0%, rgba(217,70,239,.10), transparent 60%) !important;
  background-attachment: fixed !important;}

/* main column */
#main {max-width: 900px; margin: 0 auto; width: 100%; padding: 10px 18px 14px; gap: 10px;}
#topbar {display: flex; justify-content: flex-end; padding: 8px 2px 0;}
.chip {display: inline-flex; align-items: center; gap: 7px; font-size: .78rem; font-weight: 600; padding: 5px 11px;
  border-radius: 999px; border: 1px solid var(--block-border-color); background: var(--block-background-fill);
  color: var(--body-text-color-subdued);}
.chip .dot {width: 7px; height: 7px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,.18);}
.chip .dot.loading {background: #f59e0b; box-shadow: 0 0 0 3px rgba(245,158,11,.2);}

/* chat */
#chat, #chat > div, #chat .bubble-wrap, #chat .placeholder-content {border: none !important;
  background: transparent !important; box-shadow: none !important;}
#chat {flex-grow: 1;}
#chat .message-wrap {padding-top: 6px;}
#chat .message {font-size: 15.5px; line-height: 1.62;}
#chat .user.message {background: var(--brand-grad) !important; color: #fff !important; border: none !important;}
#chat .user.message * {color: #fff !important;}
#chat .bot.message {background: var(--block-background-fill) !important; border: 1px solid var(--block-border-color) !important;}
#chat .user.message > .message, #chat .bot.message > .message {background: transparent !important; border: none !important;
  box-shadow: none !important;}
#chat button[aria-label="Clear"] {display: none !important;}  /* use the sidebar's Delete / New chat instead */
#chat pre {border-radius: 12px !important;}
#chat .placeholder-content, #chat .placeholder {width: 100%;}
.welcome {text-align: center; padding: 6vh 12px 8px;}
.welcome .logo {width: 58px; height: 58px; border-radius: 18px; margin: 0 auto 16px; display: grid; place-items: center;
  font-size: 28px; background: var(--brand-grad); box-shadow: 0 14px 34px -14px rgba(139,92,246,.9);}
.welcome h1 {font-size: clamp(1.7rem, 3.2vw, 2.3rem); font-weight: 800; letter-spacing: -.035em; margin: 0 0 6px;}
.welcome p {color: var(--body-text-color-subdued); margin: 0 auto; max-width: 560px; font-size: 1rem;}
#chat .examples {gap: 10px !important;}
#chat .example {border-radius: 14px !important; border: 1px solid var(--block-border-color) !important;
  background: var(--block-background-fill) !important; transition: transform .12s ease, border-color .12s ease;}
#chat .example:hover {transform: translateY(-2px); border-color: #8b5cf6 !important;}

/* composer */
#composer {border-radius: 22px !important; border: 1px solid var(--block-border-color) !important;
  box-shadow: 0 12px 36px -18px rgba(76,29,149,.45) !important; background: var(--block-background-fill) !important;
  padding: 4px 6px !important;}
#composer textarea {font-size: 15.5px !important;}
#composer button.submit-button, #composer button.stop-button {border-radius: 14px !important;}
#composer button.submit-button {background: var(--brand-grad) !important; color: #fff !important;}
#stats {min-height: 20px; font-size: .78rem; color: var(--body-text-color-subdued); text-align: center; padding: 0 !important;}
#stats p {margin: 0;}
.hint {text-align: center; font-size: .74rem; color: var(--body-text-color-subdued); opacity: .85;}

/* sidebar */
.sidebar {background: var(--block-background-fill) !important; border-right: 1px solid var(--block-border-color);}
.brand {display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.05rem; padding: 4px 2px 10px;}
.brand .mark {width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center; background: var(--brand-grad);
  color: #fff; font-size: 16px;}
#history, #history fieldset, #history .wrap {border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important;}
#history > span, #history [data-testid="block-info"] {background: transparent !important; padding: 0 2px 6px !important;
  font-size: .74rem !important; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued) !important;}
#history .wrap:empty::before {content: "No chats yet — your conversations will appear here.";
  font-size: .82rem; color: var(--body-text-color-subdued); padding: 2px;}
#history .wrap {flex-direction: column; gap: 4px;}
#history label {width: 100%; border-radius: 10px !important; border: none !important; padding: 9px 10px !important;
  background: transparent !important; box-shadow: none !important; font-size: .9rem;}
#history label:hover {background: var(--background-fill-secondary) !important;}
#history label.selected {background: color-mix(in srgb, #8b5cf6 16%, transparent) !important; font-weight: 600;}
#history input[type=radio] {display: none;}
#history label.selected, #history label.selected span {color: var(--body-text-color) !important;}
.sidebar .form:has(#history) {background: transparent !important; border: none !important; box-shadow: none !important;}
.side-note {font-size: .76rem; color: var(--body-text-color-subdued); line-height: 1.5;}

/* phones */
@media (max-width: 640px) {
  #main {padding: 6px 10px 10px;}
  #topbar, .hint {display: none !important;}
  .welcome {padding: 2vh 6px 4px;}
  .welcome .logo {width: 46px; height: 46px; font-size: 22px; margin-bottom: 10px;}
  .welcome h1 {font-size: 1.45rem;}
  .welcome p {font-size: .9rem;}
  #chat .examples {grid-template-columns: 1fr 1fr !important;}
  #chat .example {padding: 10px !important; font-size: .85rem !important;}
}
"""

HEAD = """
<script>
// Ctrl/⌘+K focuses the composer; Ctrl/⌘+Shift+O starts a new chat.
// Phones/small windows: start with the sidebar closed so the chat is visible. Gradio renders the sidebar
// asynchronously (later when the server is busy) and a click before its handlers attach is ignored, so keep
// checking until the sidebar is really closed, then stop (the user can still open it afterwards).
(() => {
  if (window.innerWidth >= 820) return;
  let lastClick = 0, closedChecks = 0;
  const timer = setInterval(() => {
    const sb = document.querySelector('.sidebar');
    if (!sb) return;
    if (sb.classList.contains('open')) {
      closedChecks = 0;
      if (Date.now() - lastClick > 400) { sb.querySelector('.toggle-button')?.click(); lastClick = Date.now(); }
    } else if (++closedChecks >= 3) {
      clearInterval(timer);
    }
  }, 150);
  setTimeout(() => clearInterval(timer), 30000);
})();
// While an answer is streaming, Enter must not send (it would queue and clear the text) — keep the draft.
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' || e.shiftKey || !e.target.closest || !e.target.closest('#composer')) return;
  const stop = document.querySelector('#composer .stop-button');
  if (stop && stop.offsetParent !== null) { e.preventDefault(); e.stopImmediatePropagation(); }
}, true);
document.addEventListener('keydown', (e) => {
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === 'k') {
    const t = document.querySelector('#composer textarea'); if (t) { e.preventDefault(); t.focus(); }
  }
  if (mod && e.shiftKey && e.key.toLowerCase() === 'o') {
    const b = document.querySelector('#new-chat'); if (b) { e.preventDefault(); b.click(); }
  }
});
</script>
"""


def build_theme(name: str) -> gr.themes.Base:
    """Indigo→fuchsia theme on top of the configured Gradio base theme."""
    base = {"soft": gr.themes.Soft, "monochrome": gr.themes.Monochrome, "glass": gr.themes.Glass}.get(
        name, gr.themes.Default)
    grad = "linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #d946ef 100%)"
    grad_hover = "linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #c026d3 100%)"
    return base(
        primary_hue="violet", secondary_hue="fuchsia", neutral_hue="slate", radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#f5f6fb", body_background_fill_dark="#090b14",
        background_fill_secondary="#eef0f7", background_fill_secondary_dark="#161a2e",
        block_background_fill="#ffffff", block_background_fill_dark="#11142a",
        block_border_width="1px", block_border_color="#e4e7f0", block_border_color_dark="#222744",
        block_shadow="none", block_shadow_dark="none", block_radius="16px",
        block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
        block_title_text_color="*neutral_700", block_title_text_color_dark="*neutral_200",
        block_title_text_weight="600",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_text_color="*neutral_500", block_label_text_color_dark="*neutral_400",
        block_label_border_width="0px",
        input_background_fill="#f7f8fc", input_background_fill_dark="#0b0e1f",
        input_border_color="#e1e5ef", input_border_color_dark="#262c4b",
        input_border_color_focus="*primary_400", input_border_color_focus_dark="*primary_500",
        button_primary_background_fill=grad, button_primary_background_fill_dark=grad,
        button_primary_background_fill_hover=grad_hover, button_primary_background_fill_hover_dark=grad_hover,
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        button_primary_border_color="transparent", button_primary_border_color_dark="transparent",
        button_secondary_background_fill="*neutral_100", button_secondary_background_fill_dark="#1a1f38",
        button_secondary_background_fill_hover="*neutral_200", button_secondary_background_fill_hover_dark="#232947",
        button_secondary_border_color="transparent", button_secondary_border_color_dark="#2a3052",
        slider_color="*primary_500", slider_color_dark="*primary_400",
        color_accent="*primary_500", color_accent_soft="*primary_50", color_accent_soft_dark="#1e1b3a",
    )


# ---------------------------------------------------------------------- rendering
def to_messages(turns: list[Turn], pending: Reply | None = None) -> list[dict[str, Any]]:
    """Convert stored turns (+ an in-progress reply) to Gradio chat messages."""
    out: list[dict[str, Any]] = []
    for t in turns:
        if t.role == "user":
            for img in t.images:
                if Path(img).exists():
                    out.append({"role": "user", "content": {"path": img}})
            chips = " ".join(f"`📄 {d['name']}`" for d in t.docs)
            out.append({"role": "user", "content": (chips + "\n\n" if chips else "") + t.text})
        else:
            if t.text:
                out.append({"role": "assistant", "content": t.text})
            if t.image_out and Path(t.image_out).exists():
                out.append({"role": "assistant", "content": {"path": t.image_out}})
    if pending is not None:
        body = pending.text or (f"_{pending.status}_" if pending.status else "_Thinking…_")
        out.append({"role": "assistant", "content": body})
    return out


def fmt_stats(stats: dict[str, Any], device: str) -> str:
    if not stats:
        return ""
    if stats.get("kind") == "image":
        return f"🎨 Image generated in **{stats['latency_s']:.1f}s** · {stats['model']} · {device}"
    parts = []
    if stats.get("first_token_s") is not None:
        parts.append(f"⚡ first word in **{stats['first_token_s']:.1f}s**")
    if stats.get("tokens_per_s"):
        parts.append(f"**{stats['tokens_per_s']:.0f}** tokens/s")
    if stats.get("latency_s") is not None:
        parts.append(f"{stats.get('tokens', 0)} tokens in {stats['latency_s']:.1f}s")
    parts.append(f"{stats.get('model', '')} · {device}")
    return " · ".join(parts)


def chat_title(turns: list[Turn]) -> str:
    first = next((t for t in turns if t.role == "user"), None)
    if first is None:
        return "New chat"
    title = first.text.strip().split("\n")[0]
    if not title and first.docs:
        title = first.docs[0]["name"]
    return (title[:42] + "…") if len(title) > 43 else title or "New chat"


def pick_example(evt: gr.SelectData) -> dict[str, Any]:
    return {"text": evt.value["text"], "files": []}


def _file_path(f: Any) -> str:
    return f if isinstance(f, str) else (f.get("path") or f.get("name") or "")


# ---------------------------------------------------------------------- app
class Studio:
    """Shared state (config, assistant, limiter, logger) + the Gradio UI."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        set_seed(int(cfg.app.get("seed", 42)))
        self.reg = ModelRegistry(cfg)
        self.assistant = Assistant(cfg, self.reg)
        self.limiter = RateLimiter(int(cfg.limits.get("rate_limit_per_minute", 30)))
        lg = cfg.logging
        self.events = EventLogger(lg.get("json_log_file", "outputs/logs/app.jsonl"), lg.get("level", "INFO"),
                                  bool(lg.get("log_payloads", True)), bool(lg.get("wandb", False)))
        self.max_chars = int(cfg.limits.get("max_input_chars", 8000))
        self.device_label = device_summary(self.reg.device).split(" (")[0]
        self.ready = threading.Event()

    def warmup(self) -> None:
        """Load the chat model in the background so the first message is fast."""
        def _load() -> None:
            try:
                self.assistant.warmup()
            finally:
                self.ready.set()

        threading.Thread(target=_load, daemon=True).start()

    # ------------------------------------------------------------------ handlers
    def send(self, msg: dict[str, Any] | None, turns_raw: list[dict[str, Any]], store: dict[str, Any],
             temperature: float, max_tokens: int, system_prompt: str,
             request: gr.Request) -> Iterator[tuple[Any, ...]]:
        msg = msg or {}
        text = (msg.get("text") or "").strip()
        files = [p for p in (_file_path(f) for f in msg.get("files") or []) if p]
        turns = [Turn.from_dict(t) for t in turns_raw or []]
        t0 = time.perf_counter()
        try:
            self.limiter.check(getattr(getattr(request, "client", None), "host", None) or "local")
            if text:
                text = check_length(text, self.max_chars, "Message") if len(text) > self.max_chars else text
            user = self.assistant.make_user_turn(text, files)
        except UserFacingError as exc:
            raise gr.Error(str(exc), print_exception=False) from exc

        busy = gr.MultimodalTextbox(value=None, submit_btn=False, stop_btn=True)
        yield to_messages(turns + [user], Reply(status="Thinking…")), [t.to_dict() for t in turns], gr.skip(), \
            busy, gr.skip(), gr.skip()

        reply = Reply()
        try:
            for reply in self.assistant.respond(turns, user, temperature, int(max_tokens),
                                                system_prompt.strip() or None):
                yield to_messages(turns + [user], reply), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
            status = "ok"
        except UserFacingError as exc:
            reply = Reply(text=f"⚠️ {exc}", done=True)
            status = "rejected"
        except Exception as exc:  # keep the conversation usable whatever happens
            reply = Reply(text=f"⚠️ Something went wrong: `{type(exc).__name__}: {exc}`. Please try again.",
                          done=True)
            status = "error"
            self.events.log("chat", self.assistant.model_name, "error", time.perf_counter() - t0,
                            extra={"error": repr(exc)})

        turns += [user, Turn("assistant", reply.text, image_out=reply.image)]
        turns_out = [t.to_dict() for t in turns]
        store = self._save(store, turns_out)
        if status != "error":
            self.events.log("chat", self.assistant.model_name, status, time.perf_counter() - t0,
                            inputs={"text": user.text, "files": [Path(f).name for f in files]},
                            outputs={"text": reply.text}, extra={"stats": reply.stats})
        yield (to_messages(turns), turns_out, fmt_stats(reply.stats, self.device_label),
               gr.MultimodalTextbox(**IDLE_BOX), store, self._history_update(store))

    def stop(self) -> None:
        self.assistant.engine.stop()

    def retry(self, turns_raw: list[dict[str, Any]], store: dict[str, Any], temperature: float, max_tokens: int,
              system_prompt: str, request: gr.Request) -> Iterator[tuple[Any, ...]]:
        """Regenerate the last answer."""
        turns = [Turn.from_dict(t) for t in turns_raw or []]
        if len(turns) < 2 or turns[-1].role != "assistant":
            yield to_messages(turns), turns_raw, gr.skip(), gr.skip(), gr.skip(), gr.skip()
            return
        # Re-run the stored user turn directly (images/docs are already processed — no re-upload).
        user, prior = turns[-2], turns[:-2]
        base = [t.to_dict() for t in prior]
        reply = Reply()
        yield to_messages(prior + [user], Reply(status="Regenerating…")), base, gr.skip(), \
            gr.MultimodalTextbox(submit_btn=False, stop_btn=True), gr.skip(), gr.skip()
        try:
            for reply in self.assistant.respond(prior, user, temperature, int(max_tokens),
                                                system_prompt.strip() or None):
                yield to_messages(prior + [user], reply), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        except Exception as exc:
            reply = Reply(text=f"⚠️ Something went wrong: `{type(exc).__name__}: {exc}`.", done=True)
        new = prior + [user, Turn("assistant", reply.text, image_out=reply.image)]
        out = [t.to_dict() for t in new]
        store = self._save(store, out)
        yield (to_messages(new), out, fmt_stats(reply.stats, self.device_label),
               gr.MultimodalTextbox(**IDLE_BOX), store, self._history_update(store))

    def undo(self, turns_raw: list[dict[str, Any]], store: dict[str, Any]) -> tuple[Any, ...]:
        """Remove the last exchange and put the question back in the composer for editing."""
        turns = [Turn.from_dict(t) for t in turns_raw or []]
        last_user = ""
        if turns and turns[-1].role == "assistant":
            turns.pop()
        if turns and turns[-1].role == "user":
            last_user = turns.pop().text
        out = [t.to_dict() for t in turns]
        store = self._save(store, out)
        return to_messages(turns), out, {"text": last_user, "files": []}, store, self._history_update(store)

    def new_chat(self, store: dict[str, Any]) -> tuple[Any, ...]:
        store = dict(store or {})
        store["current"] = None
        return [], [], "", gr.MultimodalTextbox(value=None, submit_btn=True, stop_btn=False), store, \
            self._history_update(store, selected=None)

    def open_chat(self, chat_id: str | None, store: dict[str, Any]) -> tuple[Any, ...]:
        store = dict(store or {})
        chat = next((c for c in store.get("items", []) if c["id"] == chat_id), None)
        if chat is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        store["current"] = chat_id
        turns = [Turn.from_dict(t) for t in chat["turns"]]
        return to_messages(turns), chat["turns"], "", store

    def delete_chat(self, store: dict[str, Any]) -> tuple[Any, ...]:
        store = dict(store or {})
        cur = store.get("current")
        store["items"] = [c for c in store.get("items", []) if c["id"] != cur]
        store["current"] = None
        return [], [], "", store, self._history_update(store, selected=None)

    # ------------------------------------------------------------------ persistence (browser localStorage)
    def _save(self, store: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
        store = dict(store or {})
        items = [dict(c) for c in store.get("items", [])]
        cur = store.get("current")
        if not turns:
            return store
        if cur is None or not any(c["id"] == cur for c in items):
            cur = uuid.uuid4().hex[:10]
            items.insert(0, {"id": cur, "title": "", "turns": []})
        for c in items:
            if c["id"] == cur:
                c["turns"] = turns
                c["title"] = chat_title([Turn.from_dict(t) for t in turns])
                c["updated"] = time.time()
        items.sort(key=lambda c: c.get("updated", 0), reverse=True)
        store["items"] = items[:MAX_SAVED_CHATS]
        store["current"] = cur
        return store

    @staticmethod
    def _history_update(store: dict[str, Any], selected: Any = "__current__") -> gr.Radio:
        items = (store or {}).get("items", [])
        sel = (store or {}).get("current") if selected == "__current__" else selected
        return gr.Radio(choices=[(c["title"] or "New chat", c["id"]) for c in items], value=sel)

    # ------------------------------------------------------------------ layout
    def build(self) -> gr.Blocks:
        title = self.cfg.app.get("title", "AI Studio")
        model = self.assistant.model_name
        welcome = f"""
<div class="welcome"><div class="logo">✦</div>
<h1>How can I help you today?</h1>
<p>Ask anything, attach a PDF, Word file, code or photo, or say <b>“draw …”</b> to create an image.
Everything runs privately on your Mac.</p></div>"""

        with gr.Blocks(title=title, fill_height=True, theme=build_theme(self.cfg.app.get("theme", "soft")),
                       css=CSS, head=HEAD) as demo:
            turns = gr.State([])
            store = gr.BrowserState({"items": [], "current": None}, storage_key="ai_studio_chats_v1")

            with gr.Sidebar(open=True, width=280, elem_classes="sidebar"):
                gr.HTML(f"<div class='brand'><div class='mark'>✦</div>{title}</div>")
                new_btn = gr.Button("＋  New chat", variant="primary", elem_id="new-chat")
                history = gr.Radio(choices=[], value=None, label="Recent chats", elem_id="history",
                                   show_label=True, container=True)
                del_btn = gr.Button("🗑  Delete this chat", size="sm", variant="secondary")
                with gr.Accordion("⚙️  Settings", open=False):
                    temperature = gr.Slider(0.0, 1.2, value=float(self.cfg.chat.get("temperature", 0.3)), step=0.05,
                                            label="Creativity", info="Low = precise & factual · High = creative")
                    max_tokens = gr.Slider(128, 4096, value=int(self.cfg.chat.get("max_tokens", 1536)), step=64,
                                           label="Max answer length (tokens)")
                    system_prompt = gr.Textbox(label="Custom instructions", lines=4,
                                               placeholder="Optional — e.g. 'Answer in Hindi' or "
                                                           "'I'm a beginner, keep it simple'. Leave empty for "
                                                           "the default assistant.")
                gr.HTML(f"""<div class="side-note">Model: <b>{model}</b><br>Running on {self.device_label} ·
                        private & offline<br><br><b>Tips</b><br>⌘K focus · ⌘⇧O new chat<br>
                        Attach files with 📎 · say “draw …” for images<br><br>Answers can be wrong —
                        double-check anything important.</div>""")

            with gr.Column(elem_id="main"):
                gr.HTML(f"<div id='topbar'><span class='chip'><span class='dot'></span>"
                        f"{model} · {self.device_label}</span></div>")
                chatbot = gr.Chatbot(
                    elem_id="chat", type="messages", show_label=False, placeholder=welcome, examples=SUGGESTIONS,
                    show_copy_button=True, layout="bubble", scale=1, height="68vh", autoscroll=True,
                    latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                      {"left": "\\(", "right": "\\)", "display": False},
                                      {"left": "\\[", "right": "\\]", "display": True}],
                    sanitize_html=True, render_markdown=True)
                box = gr.MultimodalTextbox(
                    elem_id="composer", show_label=False, file_count="multiple", file_types=FILE_TYPES,
                    sources=["upload"], placeholder="Ask anything…",
                    autofocus=True, submit_btn=True, stop_btn=False, lines=1, max_lines=10)
                stats = gr.Markdown("", elem_id="stats")
                gr.HTML("<div class='hint'>Enter to send · Shift+Enter for a new line · "
                        "Local AI can make mistakes — check important answers.</div>")

            settings = [temperature, max_tokens, system_prompt]
            outs = [chatbot, turns, stats, box, store, history]

            box.submit(self.send, [box, turns, store, *settings], outs, api_name="chat",
                       concurrency_limit=1, show_progress="hidden")
            box.stop(self.stop, None, None, api_name=False, queue=False)
            chatbot.example_select(pick_example, None, box, api_name=False).then(
                self.send, [box, turns, store, *settings], outs, api_name=False, concurrency_limit=1,
                show_progress="hidden")
            chatbot.retry(self.retry, [turns, store, *settings], outs, api_name=False, concurrency_limit=1,
                          show_progress="hidden")
            chatbot.undo(self.undo, [turns, store], [chatbot, turns, box, store, history], api_name=False)
            new_btn.click(self.new_chat, [store], [chatbot, turns, stats, box, store, history], api_name=False)
            history.input(self.open_chat, [history, store], [chatbot, turns, stats, store], api_name=False)
            del_btn.click(self.delete_chat, [store], [chatbot, turns, stats, store, history], api_name=False)
            demo.load(lambda s: self._history_update(s, selected=None), [store], [history], api_name=False)
        return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    studio = Studio(cfg)
    studio.warmup()
    demo = studio.build()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host or cfg.app.get("host", "127.0.0.1"),
        server_port=args.port or int(cfg.app.get("port", 7860)),
        share=args.share or bool(cfg.app.get("share", False)),
        allowed_paths=[str(cfg.outputs_dir)],
    )


if __name__ == "__main__":
    main()
