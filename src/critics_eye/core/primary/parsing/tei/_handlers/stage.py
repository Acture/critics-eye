from __future__ import annotations

from typing import TYPE_CHECKING, cast

import bs4

from critics_eye.core.models import (
	DocumentNode,
	KeyNodeType,
	SocialEdgeType,
	StructuralEdgeType,
	StructuralNodeType,
)

from .._labels import excerpt_display_text, fallback_display_text, tag_text
from .._models import ParseResult

if TYPE_CHECKING:
	from .._parser import TEIParser


def _attr(tag: bs4.element.Tag, key: str, default: str = "") -> str:
	value = tag.get(key, default)
	return cast(str, value)


def _attr_optional(tag: bs4.element.Tag, key: str) -> str | None:
	value = tag.get(key)
	if value is None:
		return None
	return cast(str, value)


def handle_stage(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <stage> elements (stage directions)."""
	xml_id = _attr_optional(tag, "xml:id")
	assert xml_id, f"<stage> missing xml:id: {tag}"

	# Parse n attribute for full reference: "SD 4.5.176" → (4, 5, 176)
	n_attr = _attr(tag, "n")
	n_ref = n_attr.removeprefix("SD ").removeprefix("Sd ").strip()
	parts = n_ref.split(".")
	ref_tuple: tuple[int | None, ...] = tuple(
		int(p) if p.isdigit() else None for p in parts
	)
	# Pad to at least 3 elements
	while len(ref_tuple) < 3:
		ref_tuple = (*ref_tuple, None)

	stage_node = DocumentNode(
		node_id=xml_id,
		node_reference=ref_tuple[:3],
		node_type=StructuralNodeType.STAGE_DIRECTION,
		tag="stage",
		text=tag.get_text(strip=True) or None,
		display_text=excerpt_display_text(tag.get_text(separator=" ", strip=True))
		or fallback_display_text(xml_id),
	)

	result = ParseResult()
	result.add_node(stage_node)

	# who attribute → SPOKEN_BY edges
	who_attr = _attr(tag, "who")
	for who in who_attr.split():
		char_id = who.lstrip("#")
		if char_id:
			result.add_edge(xml_id, SocialEdgeType.RELATED_TO, char_id)

	# Dispatch children (recursive stages, sounds, titles, etc.)
	child_result = parser.dispatch_children(tag)
	child_ids: list[str] = [n.node_id for n in child_result.nodes]
	result += child_result

	# NEXT edges between sequential children
	for i in range(1, len(child_ids)):
		result.add_edge(child_ids[i - 1], StructuralEdgeType.NEXT, child_ids[i])

	# CONTAINS edges from this stage to each child
	for child_id in child_ids:
		result.add_edge(xml_id, StructuralEdgeType.CONTAINS, child_id)

	return result


def handle_keyword(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <sound>, <title>, <name>, <foreign> keyword tags."""
	result = ParseResult()

	match tag.name:
		case "sound":
			type_attr = _attr(tag, "type")
			ana_attr = _attr(tag, "ana")
			node_id = f"sound_{type_attr}_{ana_attr}".strip("_")
			if not node_id or node_id == "sound":
				xml_id = _attr_optional(tag, "xml:id")
				node_id = f"sound_{xml_id or id(tag)}"
			node_type = KeyNodeType.SOUND

		case "title":
			node_id = tag.get_text(strip=True)
			node_type = KeyNodeType.TITLE

		case "name":
			node_id = tag.get_text(strip=True)
			node_type = KeyNodeType.NAME

		case "foreign":
			node_id = tag.get_text(strip=True)
			node_type = KeyNodeType.FORIEGN  # intentional typo matches enum

		case _:
			return ParseResult()

	if not node_id:
		return ParseResult()

	node = DocumentNode(
		node_id=node_id,
		node_reference=(None, None, None),
		node_type=node_type,
		tag=tag.name,
		text=tag.get_text(strip=True) or None,
		display_text=tag_text(tag) or fallback_display_text(node_id),
	)
	result.add_node(node)
	return result


def handle_seg(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <seg> elements (songs, poems, letters, etc.)."""
	seg_type = _attr(tag, "type", "unknown")
	xml_id = _attr_optional(tag, "xml:id")

	if not xml_id:
		# Synthesize from type + parent context
		parent = tag.parent
		parent_id = (
			_attr(parent, "xml:id") if isinstance(parent, bs4.element.Tag) else ""
		)
		xml_id = f"{parent_id}_seg_{seg_type}" if parent_id else f"seg_{seg_type}"

	seg_node = DocumentNode(
		node_id=xml_id,
		node_reference=(None, None, None),
		node_type=StructuralNodeType.SECTION,
		tag=f"seg_{seg_type}",
		text=None,
		display_text=seg_type,
	)

	result = ParseResult()
	result.add_node(seg_node)

	child_result = parser.dispatch_children(tag)
	result += child_result

	return result
