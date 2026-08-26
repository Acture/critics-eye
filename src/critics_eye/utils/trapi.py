from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

DEFAULT_TRAPI_URL = "http://localhost:4000/v1/chat/completions"
DEFAULT_TRAPI_MODEL = "trapi/gpt-5.2_2025-12-11"


def _render_content(content: object) -> str:
	if isinstance(content, str):
		return content
	if not isinstance(content, list):
		raise TypeError(f"Unsupported TRAPI message content: {type(content).__name__}.")
	parts: list[str] = []
	for item in content:
		if isinstance(item, str):
			parts.append(item)
		elif isinstance(item, dict):
			text = item.get("text")
			if isinstance(text, str):
				parts.append(text)
	return "\n".join(parts)


def _message_payload(
	message: BaseMessage | Mapping[str, str],
) -> dict[str, str]:
	if isinstance(message, Mapping):
		role = message.get("role")
		content = message.get("content")
		if role not in {"system", "assistant", "user"} or content is None:
			raise ValueError("TRAPI messages require a valid role and content.")
		return {"role": role, "content": content}
	if isinstance(message, SystemMessage):
		role = "system"
	elif isinstance(message, AIMessage):
		role = "assistant"
	elif isinstance(message, HumanMessage):
		role = "user"
	else:
		raise TypeError(f"Unsupported TRAPI message type: {type(message).__name__}.")
	return {"role": role, "content": _render_content(message.content)}


def normalize_trapi_model_id(model_id: str) -> str:
	return model_id if model_id.startswith("trapi/") else f"trapi/{model_id}"


class TRAPITextGenerator:
	def __init__(
		self,
		model_id: str = DEFAULT_TRAPI_MODEL,
		*,
		timeout_seconds: float = 180.0,
		timeout_retries: int = 1,
		base_url: str | None = None,
		transport: httpx.BaseTransport | None = None,
	) -> None:
		self.model_id = normalize_trapi_model_id(model_id)
		self.timeout_retries = timeout_retries
		self.base_url = base_url or os.getenv("TRAPI_BASE_URL", DEFAULT_TRAPI_URL)
		self._client = httpx.Client(
			timeout=timeout_seconds,
			transport=transport,
		)
		self._closed = False

	def invoke(
		self,
		messages: Sequence[BaseMessage | Mapping[str, str]],
		*,
		max_tokens: int | None = None,
	) -> str:
		if self._closed:
			raise RuntimeError("TRAPITextGenerator is already closed.")
		payload = {
			"model": self.model_id,
			"messages": [_message_payload(message) for message in messages],
			"temperature": 0.0,
		}
		if max_tokens is not None:
			payload["max_tokens"] = max_tokens
		for attempt in range(self.timeout_retries + 1):
			try:
				response = self._client.post(self.base_url, json=payload)
				response.raise_for_status()
			except (httpx.TransportError, httpx.HTTPStatusError):
				if attempt == self.timeout_retries:
					raise
				time.sleep(2**attempt)
				continue

			data = response.json()
			try:
				content = data["choices"][0]["message"]["content"]
			except (KeyError, IndexError, TypeError) as exc:
				raise ValueError("TRAPI returned an invalid chat-completion payload.") from exc
			if not isinstance(content, str) or not content.strip():
				raise ValueError("TRAPI returned empty message content.")
			return content
		raise RuntimeError("TRAPI retry loop exhausted.")

	def close(self) -> None:
		if self._closed:
			return
		self._closed = True
		self._client.close()

	def __enter__(self) -> TRAPITextGenerator:
		return self

	def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
		self.close()
