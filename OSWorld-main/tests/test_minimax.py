import os
import unittest
from unittest.mock import patch

from mm_agents.minimax import build_minimax_request, call_minimax


PAYLOAD = {
    "model": "MiniMax-M3",
    "messages": [
        {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
        {"role": "user", "content": [{"type": "text", "text": "Continue."}]},
    ],
    "max_tokens": 100,
    "top_p": 0.9,
    "temperature": 0.5,
}


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class MiniMaxRequestTests(unittest.TestCase):
    def test_openai_region_and_protocol_selection(self):
        url, headers, body = build_minimax_request(
            PAYLOAD, protocol="openai", region="cn_zh", api_key="test-key"
        )

        self.assertEqual(url, "https://api.minimaxi.com/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(body["model"], "MiniMax-M3")

    def test_anthropic_request_preserves_system_and_image_content(self):
        payload = {
            **PAYLOAD,
            "messages": [
                PAYLOAD["messages"][0],
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                        },
                    ],
                },
            ],
        }
        url, _, body = build_minimax_request(
            payload, protocol="anthropic", region="global_en", api_key="test-key"
        )

        self.assertEqual(url, "https://api.minimax.io/anthropic/v1/messages")
        self.assertEqual(body["system"], "Be concise.")
        self.assertEqual(body["messages"][0]["content"][1]["type"], "image")

    def test_m27_rejects_image_input(self):
        payload = {
            **PAYLOAD,
            "model": "MiniMax-M2.7",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                        }
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "text input only"):
            build_minimax_request(payload, api_key="test-key")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "MINIMAX_PROTOCOL": "openai"})
    @patch("mm_agents.minimax.requests.post")
    def test_response_text_excludes_reasoning_tags(self, post):
        post.return_value = FakeResponse(
            {"choices": [{"message": {"content": "<think>internal</think>DONE"}}]}
        )

        self.assertEqual(call_minimax(PAYLOAD), "DONE")
        self.assertEqual(post.call_args.kwargs["timeout"], 120)


if __name__ == "__main__":
    unittest.main()
