from __future__ import annotations

from collections.abc import Iterable

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from .extractor import RelationExtractor
from .model import LitAnalysis, LitAnalysisList
from .prompt import build_relation_system_prompt


def _build_repair_prompt(exc: OutputParserException) -> str:
	return (
		"Your previous output was invalid. Return only corrected JSON.\n"
		f"Parser error:\n{exc.observation}\n\n"
		f"Previous output:\n{exc.llm_output}"
	)


def extract_relations_from_text(
	text: str,
	target_works: list[str],
	extractor: RelationExtractor,
	max_retries: int = 3,
) -> list[LitAnalysis]:
	if not text.strip():
		raise ValueError("text cannot be empty")

	parser = PydanticOutputParser(pydantic_object=LitAnalysisList)
	system_prompt = build_relation_system_prompt(
		target_works,
		parser.get_format_instructions(),
	)
	messages: list[BaseMessage] = [
		SystemMessage(content=system_prompt),
		HumanMessage(
			content=f"Extract relations from the following criticism passage:\n\n{text}"
		),
	]

	for attempt in range(max_retries + 1):
		raw_output = extractor.invoke(messages)
		try:
			parsed = parser.parse(raw_output)
			return parsed.root
		except OutputParserException as exc:
			if attempt == max_retries:
				raise
			messages.append(AIMessage(content=raw_output))
			messages.append(SystemMessage(content=_build_repair_prompt(exc)))

	raise RuntimeError("Failed to parse output after all retries.")


def extract_relations_from_passages(
	passages: Iterable[str],
	target_works: list[str],
	extractor: RelationExtractor,
	max_retries: int = 3,
) -> list[LitAnalysis]:
	relations: list[LitAnalysis] = []
	for passage in passages:
		if not passage.strip():
			continue
		relations.extend(
			extract_relations_from_text(
				passage,
				target_works,
				extractor=extractor,
				max_retries=max_retries,
			)
		)
	return relations
