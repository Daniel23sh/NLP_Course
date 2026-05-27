from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request


TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
DEFAULT_OPENAI_TIMEOUT_SECONDS = 180
DEFAULT_OPENAI_MAX_ATTEMPTS = 3


def _strip_inline_comment(value: str) -> str:
    in_quote = False
    quote_char = ""
    for index, char in enumerate(value):
        if char in {"'", '"'} and (index == 0 or value[index - 1] != "\\"):
            if in_quote and char == quote_char:
                in_quote = False
                quote_char = ""
            elif not in_quote:
                in_quote = True
                quote_char = char
        if char == "#" and not in_quote:
            return value[:index].strip()
    return value.strip()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_inline_comment(value).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_default_env_files() -> None:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parent
    load_env_file(repo_root / ".env")
    load_env_file(package_root / ".env")


def build_responses_request_body(model: str, prompt: str, seed: int | None = None) -> dict:
    # The Responses API does not accept a top-level seed parameter. Keep seed in
    # the function signature so callers can stay deterministic in mock mode.
    return {
        "model": model,
        "input": prompt,
        "temperature": 0.4,
    }


class LLMClient:
    def __init__(self, mode: str = "mock", model: str = "gpt-4.1-mini") -> None:
        self.mode = mode
        self.model = model

    def _timeout_seconds(self) -> int:
        return int(os.environ.get("OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS))

    def _max_attempts(self) -> int:
        return int(os.environ.get("OPENAI_MAX_RETRIES", DEFAULT_OPENAI_MAX_ATTEMPTS))

    def _request(self, api_key: str, body: dict):
        return urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def generate(self, prompt: str, seed: int | None = None) -> str:
        if self.mode == "mock":
            return prompt
        load_default_env_files()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai mode")
        body = build_responses_request_body(self.model, prompt, seed=seed)
        max_attempts = self._max_attempts()
        timeout = self._timeout_seconds()
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with urllib.request.urlopen(self._request(api_key, body), timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in TRANSIENT_HTTP_CODES or attempt == max_attempts:
                    raise RuntimeError(f"OpenAI API request failed with HTTP {exc.code}: {error_body}") from exc
                last_error = RuntimeError(f"OpenAI transient HTTP {exc.code}: {error_body}")
            except (socket.timeout, TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    raise RuntimeError(f"OpenAI API request timed out after {max_attempts} attempts: {exc}") from exc
            time.sleep(min(2 ** (attempt - 1), 8))
        else:
            raise RuntimeError(f"OpenAI API request failed after {max_attempts} attempts: {last_error}")
        if "output_text" in payload:
            return payload["output_text"]
        chunks = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    chunks.append(content.get("text", ""))
        text = "\n".join(chunk for chunk in chunks if chunk).strip()
        if not text:
            raise RuntimeError("OpenAI response did not contain text")
        return text
