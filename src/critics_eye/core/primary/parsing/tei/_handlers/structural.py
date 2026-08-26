from __future__ import annotations

from typing import TYPE_CHECKING, cast

import bs4

from critics_eye.core.models import DocumentNode, StructuralEdgeType, StructuralNodeType

from .._labels import fallback_display_text, ref_display_text
from .._models import ParseResult

if TYPE_CHECKING:
	from .._parser import TEIParser


def _attr(tag: bs4.element.Tag, key: str, default: str = "") -> str:
	"""Get a tag attribute as a plain str, asserting it's not a multi-valued list."""
	val = tag.get(key, default)
	return cast(str, val)


def skip(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	return ParseResult()


def handle_div1(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	div_type = _attr(tag, "type")
	assert div_type, f"<div1> missing 'type' attribute: {tag}"

	match div_type:
		case "act":
			return _handle_act(tag, parser)
		case "prologue" | "epilogue" | "induction" | "preface":
			return _handle_section(tag, parser, div_type)
		case _:
			raise ValueError(f"<div1> has unrecognised type={div_type!r}: {tag}")


def handle_scene(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle a <div2> element — scene, chorus, or epilogue."""
	div2_type = _attr(tag, "type", "scene")
	n_raw = _attr(tag, "n")

	# Non-scene div2 types (chorus, epilogue): treat as SECTION
	if div2_type in ("chorus", "epilogue", "prologue"):
		return _handle_section(tag, parser, div2_type)

	# Parse scene number — may be "3" or contain non-numeric prefix like "gap"
	act_number = (
		int(_attr(tag.parent, "n", "0"))  # type: ignore[arg-type]
		if tag.parent
		and tag.parent.name == "div1"
		and _attr(tag.parent, "n", "").isdigit()  # type: ignore[arg-type]
		else 0
	)

	# Extract numeric part from n (handles "3", "2.CHO" edge cases)
	n_parts = n_raw.split(".")
	scene_number = int(n_parts[-1]) if n_parts[-1].isdigit() else 0

	scene_id = _attr(tag, "xml:id") or f"act_{act_number}_scene_{scene_number}"

	scene_node = DocumentNode(
		node_id=scene_id,
		node_reference=(act_number, scene_number, None),
		node_type=StructuralNodeType.SCENE,
		tag="scene",
		text=None,
		display_text=ref_display_text("Scene", (act_number, scene_number))
		or fallback_display_text(scene_id),
	)

	result = ParseResult()
	result.add_node(scene_node)

	child_ids: list[str] = []

	for child in tag.children:
		if not isinstance(child, bs4.element.Tag):
			continue
		child_tag = child.name
		match child_tag:
			case "head" | "lb" | "pb" | "fw" | "label":
				continue
			case "milestone":
				assert (
					child.get("unit") == "page"
				), f"<milestone> with unexpected unit in scene {scene_id}: {child}"
			case "sp" | "stage":
				child_result = parser.dispatch(child)
				if child_result.nodes:
					child_ids.append(child_result.nodes[0].node_id)
				result += child_result
			case _:
				raise ValueError(
					f"Unexpected tag <{child_tag}> inside <div2> scene {scene_id}"
				)

	# NEXT edges between sequential section-level children (speech→speech, stage→speech, etc.)
	for i in range(1, len(child_ids)):
		result.add_edge(child_ids[i - 1], StructuralEdgeType.NEXT, child_ids[i])

	# CONTAINS edges from scene to each child
	for child_id in child_ids:
		result.add_edge(scene_id, StructuralEdgeType.CONTAINS, child_id)

	return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _handle_act(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	act_number = int(_attr(tag, "n"))
	act_id = _attr(tag, "xml:id") or f"act_{act_number}"

	act_node = DocumentNode(
		node_id=act_id,
		node_reference=(act_number, None, None),
		node_type=StructuralNodeType.ACT,
		tag="act",
		text=None,
		display_text=ref_display_text("Act", (act_number,))
		or fallback_display_text(act_id),
	)

	result = ParseResult()
	result.add_node(act_node)

	scene_ids: list[str] = []
	for div2 in tag.find_all("div2", recursive=False):
		scene_result = parser.dispatch(div2)
		if scene_result.nodes:
			scene_ids.append(scene_result.nodes[0].node_id)
		result += scene_result

	# NEXT edges between sequential scenes
	for i in range(1, len(scene_ids)):
		result.add_edge(scene_ids[i - 1], StructuralEdgeType.NEXT, scene_ids[i])

	# CONTAINS edges from act to each scene
	for scene_id in scene_ids:
		result.add_edge(act_id, StructuralEdgeType.CONTAINS, scene_id)

	return result


def _handle_section(
	tag: bs4.element.Tag, parser: TEIParser, section_type: str
) -> ParseResult:
	section_id = _attr(tag, "xml:id") or section_type

	section_node = DocumentNode(
		node_id=section_id,
		node_reference=(None, None, None),
		node_type=StructuralNodeType.SECTION,
		tag=section_type,
		text=None,
		display_text=section_type,
	)

	result = ParseResult()
	result.add_node(section_node)

	child_result = parser.dispatch_children(tag)
	child_ids: list[str] = [n.node_id for n in child_result.nodes]
	result += child_result

	# NEXT edges between sequential children
	for i in range(1, len(child_ids)):
		result.add_edge(child_ids[i - 1], StructuralEdgeType.NEXT, child_ids[i])

	# CONTAINS edges from section to each child
	for child_id in child_ids:
		result.add_edge(section_id, StructuralEdgeType.CONTAINS, child_id)

	return result
