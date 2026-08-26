from __future__ import annotations

from typing import TYPE_CHECKING

import bs4

from critics_eye.core.models import DocumentNode, StructuralEdgeType, StructuralNodeType

from .._labels import excerpt_display_text, fallback_display_text, ref_display_text, tag_text
from .._models import ParseResult
from .structural import IGNORED_CHILDREN, _attr, _attr_optional

if TYPE_CHECKING:
	from .._parser import DraCorTEIParser


def _parse_ref(ref_str: str) -> tuple[int | None, ...]:
	parts = ref_str.split(".")
	values: list[int | None] = []
	for part in parts:
		part = part.strip()
		values.append(int(part) if part.isdigit() else None)
	while len(values) < 3:
		values.append(None)
	return tuple(values[:3])


def _direct_text(tag: bs4.element.Tag) -> str | None:
	text = tag.get_text(separator=" ", strip=True)
	return " ".join(text.split()) or None


def handle_speech(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	result = ParseResult()
	speech_id = _attr_optional(tag, "xml:id") or parser.next_synth_id("sp")
	who_attr = _attr(tag, "who")
	first_ref = (None, None, None)
	first_line = tag.find(["l", "p"], recursive=True)
	if isinstance(first_line, bs4.element.Tag):
		n_attr = _attr_optional(first_line, "n")
		if n_attr:
			first_ref = _parse_ref(n_attr)

	result.add_node(
		DocumentNode(
			node_id=speech_id,
			node_reference=first_ref,
			node_type=StructuralNodeType.SECTION,
			tag="speech",
			text=None,
			display_text=ref_display_text("Speech", first_ref) or fallback_display_text(speech_id),
		)
	)

	for raw in who_attr.split():
		speaker_id = raw.lstrip("#")
		if speaker_id:
			result.add_edge(speech_id, StructuralEdgeType.SPOKEN_BY, speaker_id)

	child_ids: list[str] = []
	line_like_children = [
		child for child in tag.children if isinstance(child, bs4.element.Tag) and child.name in {"l", "p", "lg", "stage"}
	]
	for child in line_like_children:
		child_result = parser.dispatch(child)
		if child_result.nodes:
			child_ids.append(child_result.nodes[0].node_id)
		result += child_result

	for child_id in child_ids:
		result.add_edge(speech_id, StructuralEdgeType.CONTAINS, child_id)
	return result


def handle_line(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	text = _direct_text(tag)
	if text is None:
		return ParseResult()
	line_id = _attr_optional(tag, "xml:id") or parser.next_synth_id(tag.name)
	n_attr = _attr_optional(tag, "n")
	ref = _parse_ref(n_attr) if n_attr else (None, None, None)
	result = ParseResult()
	result.add_node(
		DocumentNode(
			node_id=line_id,
			node_reference=ref,
			node_type=StructuralNodeType.LINE,
			tag="line",
			text=text,
			display_text=excerpt_display_text(text),
		)
	)
	result.line_ids.append(line_id)
	return result


def handle_line_group(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	result = ParseResult()
	for child in tag.children:
		if not isinstance(child, bs4.element.Tag):
			continue
		match child.name:
			case "l" | "p" | "stage" | "lg":
				result += parser.dispatch(child)
			case "head" | "pb":
				continue
			case _ if child.name in IGNORED_CHILDREN:
				continue
			case _:
				raise ValueError(f"Unexpected tag <{child.name}> inside <lg>")
	return result


def skip_speaker(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	return ParseResult()
