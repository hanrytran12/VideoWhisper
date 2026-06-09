"""Minimal Ollama HTTP client for VLM image description, with on-disk cache."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import requests


class OllamaVLM:
    def __init__(
        self,
        host: str,
        model: str,
        prompt: str,
        timeout: int = 120,
        num_gpu: int | None = None,
        cache_path: Path | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.cache_path = cache_path
        self._cache: dict[str, str] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    @staticmethod
    def _hash_image(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def describe(self, image_path: Path) -> str:
        image_bytes = Path(image_path).read_bytes()
        key = self._hash_image(image_bytes)
        if key in self._cache:
            return self._cache[key]

        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload: dict = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [b64],
            "stream": False,
        }
        if self.num_gpu is not None:
            payload["options"] = {"num_gpu": self.num_gpu}

        resp = requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        self._cache[key] = text
        self._save_cache()
        return text

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.ok
        except Exception:
            return False
