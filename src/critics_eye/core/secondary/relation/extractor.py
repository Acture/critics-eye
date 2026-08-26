from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage

from critics_eye.utils.trapi import DEFAULT_TRAPI_MODEL, TRAPITextGenerator


DEFAULT_LLM_MODEL = DEFAULT_TRAPI_MODEL


class RelationExtractor(Protocol):
	def invoke(self, messages: Sequence[BaseMessage]) -> str: ...
	def close(self) -> None: ...


class TRAPIRelationExtractor:
	def __init__(
		self,
		model_id: str = DEFAULT_LLM_MODEL,
		*,
		timeout_seconds: float = 180.0,
		timeout_retries: int = 1,
	) -> None:
		self._generator = TRAPITextGenerator(
			model_id=model_id,
			timeout_seconds=timeout_seconds,
			timeout_retries=timeout_retries,
		)

	def invoke(self, messages: Sequence[BaseMessage]) -> str:
		return self._generator.invoke(messages)

	def close(self) -> None:
		self._generator.close()
