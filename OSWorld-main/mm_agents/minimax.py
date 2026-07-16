"""MiniMax-compatible chat adapters for the prompt-based desktop agent."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

import requests


MINIMAX_MODELS = frozenset({"MiniMax-M3", "MiniMax-M2.7"})
MINIMAX_TEXT_ONLY_MODELS = frozenset({"MiniMax-M2.7"})
MINIMAX_ENDPOINTS = {
    "global_en": {
        "openai": "https://api.minimax.io/v1",
        "anthropic": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai": "https://api.minimaxi.com/v1",
        "anthropic": "https://api.minimaxi.com/anthropic",
    },
}
MINIMAX_PROTOCOLS = frozenset({"openai", "anthropic"})


def _content_parts(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    content = message.get("content", [])
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _contains_non_text(messages: list[Mapping[str, Any]]) -> bool:
    return any(
        part.get("type") not in {"text"}
        for message in messages
        for part in _content_parts(message)
    )


def _to_anthropic_content(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = []
    for part in _content_parts(message):
        part_type = part.get("type")
        if part_type == "text":
            content.append({"type": "text", "text": part["text"]})
        elif part_type == "image_url":
            image_url = part["image_url"]["url"]
            if image_url.startswith("data:"):
                header, image_data = image_url.split(",", 1)
                media_type = header[5:].split(";", 1)[0] or "image/png"
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    }
                )
            else:
                content.append(
                    {"type": "image", "source": {"type": "url", "url": image_url}}
                )
        else:
            raise ValueError(f"Unsupported MiniMax content type: {part_type}")
    return content


def _selected_protocol(protocol: str | None) -> str:
    selected = protocol or os.environ.get("MINIMAX_PROTOCOL", "openai")
    if selected not in MINIMAX_PROTOCOLS:
        raise ValueError("MINIMAX_PROTOCOL must be 'openai' or 'anthropic'")
    return selected


def _selected_region(region: str | None) -> str:
    selected = region or os.environ.get("MINIMAX_REGION", "global_en")
    if selected not in MINIMAX_ENDPOINTS:
        raise ValueError("MINIMAX_REGION must be 'global_en' or 'cn_zh'")
    return selected


def _thinking_parameter(model: str) -> dict[str, str] | None:
    thinking = os.environ.get("MINIMAX_THINKING")
    if not thinking:
        return None
    if model != "MiniMax-M3":
        raise ValueError("MINIMAX_THINKING only applies to MiniMax-M3")
    if thinking not in {"adaptive", "disabled"}:
        raise ValueError("MINIMAX_THINKING must be 'adaptive' or 'disabled'")
    return {"type": thinking}


def build_minimax_request(
    payload: Mapping[str, Any],
    *,
    protocol: str | None = None,
    region: str | None = None,
    api_key: str | None = None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build a request while keeping protocol and region selection user-configurable."""
    model = payload["model"]
    if model not in MINIMAX_MODELS:
        raise ValueError(f"Unsupported MiniMax model: {model}")

    messages = list(payload["messages"])
    if model in MINIMAX_TEXT_ONLY_MODELS and _contains_non_text(messages):
        raise ValueError(f"{model} supports text input only")

    selected_protocol = _selected_protocol(protocol)
    selected_region = _selected_region(region)
    api_key = api_key or os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY must be set")

    base_url = MINIMAX_ENDPOINTS[selected_region][selected_protocol]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    thinking = _thinking_parameter(model)

    if selected_protocol == "openai":
        request_body = dict(payload)
        if thinking:
            request_body["thinking"] = thinking
        return f"{base_url}/chat/completions", headers, request_body

    request_messages = []
    system_parts = []
    for message in messages:
        if message["role"] == "system":
            system_parts.extend(
                part["text"] for part in _content_parts(message) if part.get("type") == "text"
            )
            continue
        request_messages.append(
            {"role": message["role"], "content": _to_anthropic_content(message)}
        )

    request_body = {
        "model": model,
        "max_tokens": payload["max_tokens"],
        "messages": request_messages,
        "temperature": payload["temperature"],
        "top_p": payload["top_p"],
    }
    if system_parts:
        request_body["system"] = "\n\n".join(system_parts)
    if thinking:
        request_body["thinking"] = thinking
    return f"{base_url}/v1/messages", headers, request_body


def _text_response(content: Any) -> str:
    if isinstance(content, str):
        text = content
    else:
        text = "\n".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text" and block.get("text")
        )
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def call_minimax(payload: Mapping[str, Any]) -> str:
    """Send a non-streaming request and return only assistant text."""
    protocol = _selected_protocol(None)
    request_url, headers, request_body = build_minimax_request(payload, protocol=protocol)
    response = requests.post(request_url, headers=headers, json=request_body, timeout=120)
    response.raise_for_status()
    data = response.json()
    if protocol == "openai":
        return _text_response(data["choices"][0]["message"]["content"])
    return _text_response(data["content"])
