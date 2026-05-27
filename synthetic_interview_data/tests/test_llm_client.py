import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm_client import LLMClient, build_responses_request_body, load_env_file


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


class LLMClientEnvTests(unittest.TestCase):
    def test_load_env_file_sets_missing_values_without_overwriting_existing_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                'OPENAI_API_KEY="from_file"\nEXISTING_VALUE=from_file\nCOMMENTED=nope # inline comment\n',
                encoding="utf-8",
            )
            os.environ["EXISTING_VALUE"] = "from_environment"
            try:
                load_env_file(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "from_file")
                self.assertEqual(os.environ["EXISTING_VALUE"], "from_environment")
                self.assertEqual(os.environ["COMMENTED"], "nope")
            finally:
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("EXISTING_VALUE", None)
                os.environ.pop("COMMENTED", None)

    def test_responses_request_body_does_not_send_seed(self):
        body = build_responses_request_body("gpt-4.1-mini", "hello", seed=123)

        self.assertEqual(body["model"], "gpt-4.1-mini")
        self.assertEqual(body["input"], "hello")
        self.assertNotIn("seed", body)

    def test_openai_generate_retries_timeout_then_returns_text(self):
        calls = [socket.timeout("slow"), FakeResponse(b'{"output_text":"ok"}')]
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            with patch("src.llm_client.urllib.request.urlopen", side_effect=calls) as urlopen:
                with patch("src.llm_client.time.sleep") as sleep:
                    text = LLMClient(mode="openai", model="gpt-4.1-mini").generate("prompt")

            self.assertEqual(text, "ok")
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once()
        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_openai_generate_raises_clear_error_after_repeated_timeouts(self):
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            with patch("src.llm_client.urllib.request.urlopen", side_effect=socket.timeout("slow")):
                with patch("src.llm_client.time.sleep"):
                    with self.assertRaisesRegex(RuntimeError, "timed out after 3 attempts"):
                        LLMClient(mode="openai", model="gpt-4.1-mini").generate("prompt")
        finally:
            os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
