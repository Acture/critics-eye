from __future__ import annotations

import json
import logging
from string import Template

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, RootModel

from critics_eye.core.models import SocialEdgeType
from critics_eye.core.models.node_type import KeyNodeType
from critics_eye.utils.trapi import DEFAULT_TRAPI_MODEL, TRAPITextGenerator
from ._models import ParseResult

allowed_types_str = ", ".join([f"'{e.value}'" for e in SocialEdgeType])

SOCIAL_RELATION_PROMPT = Template("""
You are an expert graph database engineer and literary analyst.
Your task is to extract social relationships between characters based strictly on their textual descriptions.

**CRITICAL PREREQUISITES - IGNORE IF NOT MET**:
1.  **Must Be Explicit**: The description must explicitly state or strongly imply a relationship (e.g., "son of", "servant to"). Do not guess based on external knowledge not present in the provided text.
2.  **Must Use Provided IDs**: The `source_id` and `target_id` MUST be exact strings from the input JSON keys. **DO NOT** normalize IDs (e.g., use "Macbeth_Mac", NOT "Macbeth").
3.  **Must Be Allowed Type**: Use ONLY these lowercase strings for `relation_type`: $allowed_types_str.

**Extraction Logic**:
* **Inference**:
    * "X is son of Y" -> X `kin_of` Y
    * "X is attendant to Y" -> X `serves` Y
    * "X is wife of Y" -> X `kin_of` Y
* **Direction**: For `serves`, strictly follow the logical direction. For `kin_of`, `ally_of`, and `enemy_of`, output one edge without trying to encode subtypes beyond the allowed label set.

**Output JSON format**:
You must output a single valid JSON list.
[
  {{
    "source_id": "...", // Exact key from input
    "target_id": "...", // Exact key from input
    "relation_type": "...", // One of the allowed types
    "reasoning": "..." // Quote or brief explanation
  }}
]

**Example—Desired Extraction**:
Input Nodes:
{{
  "Seyton_Mac": "Seyton, attendant to Macbeth",
  "Macbeth_Mac": "King of Scotland"
}}
Output:
[
  {{
    "source_id": "Seyton_Mac",
    "target_id": "Macbeth_Mac",
    "relation_type": "serves",
    "reasoning": "Text describes Seyton as 'attendant to Macbeth'."
  }}
]

**Now, analyze the following nodes strictly:**

$format_instructions
""")


class SocialRelationship(BaseModel):
	"""Represents a single social edge extracted from text."""

	source_id: str = Field(
		..., description="The exact ID of the source character from the input keys."
	)
	target_id: str = Field(
		..., description="The exact ID of the target character from the input keys."
	)
	relation_type: SocialEdgeType | str = Field(
		..., description="The specific type of social relationship."
	)
	reasoning: str = Field(
		..., description="Brief text quote or explanation justifying this relation."
	)


class ExtractedRelationships(RootModel[list[SocialRelationship]]):
	root: list[SocialRelationship]


def _find_social_relations(ret: ParseResult) -> list[SocialRelationship]:
	character_info: dict[str, str] = {}

	for node in ret.nodes:
		# 过滤条件：必须是 Keyword 且 tag 包含 person
		# 注意：根据你的 JSON，node_type 可能是字符串 "keyword" 也可能是 Enum
		if node.node_type in {KeyNodeType.CHARACTER, KeyNodeType.GROUP}:
			character_info[node.node_id] = (node.text or "").strip()
		else:
			logging.info(
				f"Skipping node {node.node_id} with type {node.node_type} for social relation extraction"
			)

	if not character_info:
		logging.info("没有找到带有描述的人物节点。")
		return []

	parser = PydanticOutputParser(pydantic_object=ExtractedRelationships)

	system_msg_content = SOCIAL_RELATION_PROMPT.substitute(
		allowed_types_str=allowed_types_str,
		format_instructions=parser.get_format_instructions(),
	)

	input_json_str = json.dumps(character_info, indent=2, ensure_ascii=False)

	base_messages: list[SystemMessage | HumanMessage | AIMessage] = [
		SystemMessage(content=system_msg_content),
		HumanMessage(content=f"Extract relations from these nodes:\n{input_json_str}"),
	]

	max_retries = 5
	current_messages = base_messages.copy()

	with TRAPITextGenerator(
		model_id=DEFAULT_TRAPI_MODEL,
		timeout_seconds=120.0,
	) as generator:
		for attempt in range(max_retries + 1):
			try:
				raw_output = generator.invoke(current_messages)
				return parser.parse(raw_output).root

			except OutputParserException as exc:
				if attempt == max_retries:
					raise

				bad_output = str(getattr(exc, "llm_output", ""))
				current_messages.append(AIMessage(content=bad_output))
				current_messages.append(
					HumanMessage(
						content=(
							f"Validation Error: {exc}\n\n"
							"You must output a raw JSON list: [ {...}, {...} ]\n"
							f"Make sure `relation_type` is one of: {allowed_types_str}\n"
							"Fix the JSON."
						)
					)
				)

	raise RuntimeError("Failed to parse social relations after all retries.")
