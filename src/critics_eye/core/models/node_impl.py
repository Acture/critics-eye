from __future__ import annotations  # 确保类型提示向前兼容

from typing import Annotated, Optional, Tuple

from pydantic import AfterValidator, BaseModel, Field

from .node_type import NodeType


def is_not_none(v: Optional[str]):
	if v is None or v == "":
		raise ValueError("Cannot be None")
	return v


class DocumentNode(BaseModel):
	"""
	代表文本文档中的一个结构节点。
	"""

	node_id: Annotated[str, AfterValidator(is_not_none)]
	node_reference: Optional[Tuple[Optional[int], ...]]
	node_type: Annotated[NodeType, AfterValidator(is_not_none)]

	text: Optional[str] = Field(None, description="此节点的文本内容 (如果它是叶子节点)")
	display_text: Optional[str] = Field(
		None, description="节点的人类可读标签，不要求对应正文文本。"
	)

	tag: Optional[str] = Field(None)

	# 3. 哈希和相等性 (不变)
	def __hash__(self):
		return hash(self.node_id)

	def __eq__(self, other):
		if not isinstance(other, DocumentNode):
			return NotImplemented
		return self.node_id == other.node_id
