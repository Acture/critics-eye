from __future__ import annotations

import json

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from critics_eye.utils.trapi import TRAPITextGenerator


def test_trapi_text_generator_uses_chat_completion_contract() -> None:
	def handler(request: httpx.Request) -> httpx.Response:
		payload = json.loads(request.content)
		assert payload == {
			"model": "trapi/test-model",
			"messages": [
				{"role": "system", "content": "Follow instructions."},
				{"role": "user", "content": "Reply OK."},
			],
			"temperature": 0.0,
			"max_tokens": 20,
		}
		return httpx.Response(
			200,
			json={"choices": [{"message": {"content": "OK"}}]},
		)

	with TRAPITextGenerator(
		model_id="test-model",
		base_url="http://trapi.test/v1/chat/completions",
		transport=httpx.MockTransport(handler),
	) as generator:
		result = generator.invoke(
			[
				SystemMessage(content="Follow instructions."),
				HumanMessage(content="Reply OK."),
			],
			max_tokens=20,
		)

	assert result == "OK"
