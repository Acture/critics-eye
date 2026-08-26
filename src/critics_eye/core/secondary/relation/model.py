from pathlib import Path
from typing import Annotated, List, Optional, Tuple

from pydantic import BaseModel, Field, RootModel, WithJsonSchema, field_validator

from critics_eye.core.models.edge import RelationLabel
from critics_eye.core.secondary.relation.citation import parse_citation


# 模型逐字抄录引用字符串，由 citation.parse_citation 确定性解析为三元组。
# 让模型自己做罗马数字换算会同时带来幻觉和偷懒：在半数引用带行号的语料上，
# 它只给出 20% 的行号，把 `III. iii. 106` 悄悄降格成场级引用。
#
# Python 侧仍是三元组，下游与既有 JSONL 都不受影响；WithJsonSchema 只改对模型
# 公布的那份 schema。少了它，PydanticOutputParser 会把 `prefixItems: [integer]`
# 注入提示词，与「抄录字符串」的指令直接冲突，模型服从 schema 而放弃内容——
# 实测整份抽取的引用全部退化为 [null, null, null]。
CitationRef = Annotated[
	Tuple[Optional[int], Optional[int], Optional[int]],
	WithJsonSchema(
		{
			"type": ["string", "null"],
			"description": "Citation copied verbatim, e.g. 'III. iii. 106', '(2.1.33)', 'V. ii.'; null if absent.",
		},
		mode="validation",
	),
]


class LitAnalysis(BaseModel):
	source_element: str = Field(
		...,
		description=(
			"关系中的源头文学元素、主题或段落核心思想。"
			"The source literary element, theme, or core idea of a paragraph in the relationship."
		),
	)
	target_element: str = Field(
		...,
		description=(
			"关系中的目标文学元素、主题或段落核心思想。"
			"The target literary element, theme, or core idea of a paragraph in the relationship."
		),
	)
	relationship_type: List[RelationLabel | str] = Field(
		...,
		description=(
			"元素间的关系类型（例如，象征、平行、对比、意图）。"
			"The type of relationship between the elements (e.g., symbolizes, parallels, contrasts, intends)."
		),
	)
	source_citation: CitationRef = Field(
		...,
		description=(
			"源元素的引用，按批评原文逐字抄录，例如 'III. iii. 106'、'(2.1.33)' 或 'V. ii.'。"
			"Copy it exactly as the critic wrote it; do not convert or simplify it."
			"如果该元素没有引用，使用 null。"
		),
	)

	target_citation: CitationRef = Field(
		...,
		description=(
			"目标元素的引用，按批评原文逐字抄录，例如 'V. ii.'、'[5.1.35]' 或 'Act III, Scene 2'。"
			"Copy it exactly as the critic wrote it; do not convert or simplify it."
			"如果该元素没有引用，使用 null。"
		),
	)

	@field_validator("source_citation", "target_citation", mode="before")
	@classmethod
	def parse_citation_string(cls, v: object) -> object:
		"""Turn a verbatim citation into `(act, scene, line)`.

		Strings are what the model is now asked for. Lists are still accepted so
		that JSONL written by earlier runs keeps loading.
		"""
		if isinstance(v, str):
			return parse_citation(v)
		if v is None:
			return (None, None, None)
		return v
	justification_quote: str = Field(
		...,
		description=(
			"支持该关系的批评文本中的引文。"
			"The quote from the critical text justifying the relationship."
		),
	)
	confidence: float = Field(
		...,
		ge=0.0,
		le=1.0,
		description=(
			"关系陈述的明确程度置信度（0.0 到 1.0）。"
			"Confidence level for how explicitly the relationship is stated (0.0 to 1.0)."
		),
	)
	source_file: Optional[str] = Field(None, description=("关系的原始批评文本文件。"))
	play: Optional[str] = Field(
		None,
		description=(
			"该批评所讨论的剧目代号，例如 'Mac'。"
			"A citation is only (act, scene, line), so once reviews from several "
			"plays are merged there is nothing else to tell them apart: Macbeth's "
			"1.1.3 resolves against Hamlet's 1.1.3 just as well. Set from the "
			"per-play review file it was read from, never by the model."
		),
	)

	@field_validator("relationship_type", mode="before")
	@classmethod
	def clean_relationship_types(cls, v: object) -> object:
		"""
		在验证前自动“清理” LLM 的输出。
		1. 确保它是一个列表。
		2. 将所有字符串成员转换为小写。
		"""
		if not isinstance(v, list):
			return v  # 让它在标准验证中失败

		# 遍历列表，将所有字符串转为小写
		cleaned_list = []
		for item in v:
			if isinstance(item, str):
				cleaned_list.append(item.lower())
			else:
				cleaned_list.append(item)  # 保留非字符串（如果有的话）

		return cleaned_list


class LitAnalysisList(RootModel[List[LitAnalysis]]):
	@staticmethod
	def from_jsonline(p: Path) -> "LitAnalysisList":
		lines = p.read_text().splitlines()
		d = [LitAnalysis.model_validate_json(line) for line in lines]
		return LitAnalysisList(d)
