"""Input limits, image-prompt safety filter and a per-client rate limiter."""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import defaultdict, deque


class UserFacingError(Exception):
    """An error whose message is safe and helpful to show in the UI."""


# Deliberately conservative keyword list; a production system should add a
# learned classifier (e.g. the diffusers safety checker) on top.
_BLOCKED_TERMS = [
    "nude", "nudity", "naked", "nsfw", "porn", "explicit sex", "sexual", "genitals",
    "gore", "beheading", "dismember", "mutilat", "child abuse", "csam", "underage",
    "self-harm", "suicide", "terrorist attack", "bomb making", "swastika",
]
_BLOCKED_RE = re.compile("|".join(re.escape(t) for t in _BLOCKED_TERMS), re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize unicode/quotes and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def check_length(text: str, max_chars: int, field: str = "Input") -> str:
    """Validate non-empty, length-bounded input and return the cleaned text."""
    text = clean_text(text)
    if not text:
        raise UserFacingError(f"{field} is empty — please enter some text.")
    if max_chars and len(text) > max_chars:
        raise UserFacingError(
            f"{field} is {len(text):,} characters; the limit is {max_chars:,}. Please shorten it."
        )
    return text


def check_image_prompt(prompt: str) -> None:
    """Reject prompts that match the unsafe-content blocklist."""
    m = _BLOCKED_RE.search(prompt)
    if m:
        raise UserFacingError(
            f"This prompt was blocked by the content filter (matched '{m.group(0)}'). "
            "Please rephrase it."
        )


class RateLimiter:
    """Sliding-window limiter: at most `per_minute` calls per key per 60 s."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                wait = int(60 - (now - q[0])) + 1
                raise UserFacingError(f"Rate limit reached — please wait ~{wait}s and try again.")
            q.append(now)
