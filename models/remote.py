"""Minimal OpenAI-compatible chat-completions client for `backend: remote` tasks.

Works with OpenAI and any compatible gateway (Vertex/Bedrock proxies, vLLM, Ollama, LiteLLM)
by changing `remote.base_url` in configs/app.yaml.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from core.safety import UserFacingError


class RemoteClient:
    def __init__(self, base_url: str, api_key_env: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise UserFacingError(
                f"Remote backend selected but ${self.api_key_env} is not set. "
                "Export the key or switch the task backend to 'hf' in configs/app.yaml."
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise UserFacingError(f"Remote endpoint error {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise UserFacingError(f"Could not reach remote endpoint: {exc.reason}") from exc
        return data["choices"][0]["message"]["content"].strip()
