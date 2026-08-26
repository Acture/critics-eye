from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from critics_eye.core.secondary.motif.model import LabeledMotifCluster, MotifCluster
from critics_eye.utils.trapi import DEFAULT_TRAPI_MODEL, TRAPITextGenerator
from critics_eye.utils.progress import iter_with_progress

DEFAULT_LLM_MODEL = DEFAULT_TRAPI_MODEL


class LabelResult(BaseModel):
	short_labels: list[str]


def generate_labels(
	data_path: Path,
	output_path: Path,
	model_id: str = DEFAULT_LLM_MODEL,
) -> None:
	json_data = json.loads(data_path.read_text(encoding="utf-8"))
	data = [MotifCluster.model_validate(d) for d in json_data]
	parser = PydanticOutputParser(pydantic_object=LabelResult)

	prompt_template = Template("""
	Task: Summarize the following literary motif cluster into a concise label (2-5 words).

	**HERE ARE THE STRICT OUTPUT FORMATTING INSTRUCTIONS YOU MUST FOLLOW:**
	$format_instructions
	""")
	system_prompt = prompt_template.substitute(
		format_instructions=parser.get_format_instructions()
	)

	result: list[LabeledMotifCluster] = []
	max_retries = 3

	logging.info("Generating labels for %s motifs with %s...", len(data), model_id)
	with TRAPITextGenerator(model_id=model_id, timeout_seconds=120.0) as generator:
		for raw_motif in iter_with_progress(
			data,
			description="Labeling motifs",
			total=len(data),
		):
			messages: list[SystemMessage | HumanMessage | AIMessage] = [
				SystemMessage(content=system_prompt),
				HumanMessage(
					content=(
						"The following are the members of this motif cluster: "
						f"{raw_motif.model_dump_json()}"
					)
				),
			]
			for attempt in range(max_retries + 1):
				try:
					raw_output = generator.invoke(messages)
					parsed = parser.parse(raw_output)
					result.append(
						LabeledMotifCluster(
							**raw_motif.model_dump(),
							labels=parsed.short_labels,
						)
					)
					break
				except OutputParserException as exc:
					if attempt == max_retries:
						raise
					fix_prompt = (
						"Your last output was invalid. Please fix it.\n"
						f"Here is the error:\n{exc.observation}\n\n"
						"Here is the bad output you provided:\n"
						f"{exc.llm_output}"
					)
					messages.append(AIMessage(content=exc.llm_output))
					messages.append(SystemMessage(content=fix_prompt))

	output_path.write_text(
		json.dumps([r.model_dump() for r in result], indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	logging.info("Labels saved to %s", output_path)
