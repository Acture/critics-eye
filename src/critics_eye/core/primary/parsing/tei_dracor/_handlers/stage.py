from __future__ import annotations

from typing import TYPE_CHECKING

import bs4

from critics_eye.core.models import DocumentNode, StructuralNodeType

from .._labels import excerpt_display_text, fallback_display_text
from .._models import ParseResult
from .structural import _attr_optional

if TYPE_CHECKING:
	from .._parser import DraCorTEIParser


def handle_stage(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	text = " ".join(tag.get_text(separator=" ", strip=True).split()) or None
	if text is None:
		return ParseResult()
	stage_id = _attr_optional(tag, "xml:id") or parser.next_synth_id("stage")
	result = ParseResult()
	result.add_node(
		DocumentNode(
			node_id=stage_id,
			node_reference=(None, None, None),
			node_type=StructuralNodeType.STAGE_DIRECTION,
			tag="stage",
			text=text,
			display_text=excerpt_display_text(text) or fallback_display_text(stage_id),
		)
	)
	return result
