from __future__ import annotations

from typing import TYPE_CHECKING, cast

import bs4

from critics_eye.core.models import DocumentNode, StructuralEdgeType, StructuralNodeType

from .._labels import fallback_display_text, ref_display_text, tag_text
from .._models import ParseResult
from .._ordinals import parse_german_ordinal

if TYPE_CHECKING:
	from .._parser import DraCorTEIParser


def _attr(tag: bs4.element.Tag, key: str, default: str = "") -> str:
	value = tag.get(key, default)
	return cast(str, value)


def _attr_optional(tag: bs4.element.Tag, key: str) -> str | None:
	value = tag.get(key)
	if value is None:
		return None
	return cast(str, value)


def skip(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	return ParseResult()


# Verse drama places choruses and songs as bare <lg> directly under a scene, and
# prose drama does the same with <p>; neither is wrapped in <sp> because they
# have no speaker. Acts, scenes and untyped sections therefore accept the same
# child set -- restricting acts/scenes to {div, sp, stage} rejected Faust,
# Nathan der Weise and Kabale und Liebe outright.
CONTAINER_CHILDREN = {"div", "sp", "stage", "l", "p", "lg"}

# Front and back matter that Early Modern English editions attach to a scene:
# a dedication signature, a scene argument, an inset letter or song. These carry
# text, but not text any character speaks on stage, so counting them as sections
# would inflate the corpus with material that has no (act, scene, line) address.
#
# <floatingText> is the biggest single case (103 of 285 English plays): a
# play-within-a-play or an interpolated masque, structurally a whole nested TEI
# text. It needs its own traversal to be represented properly; ignoring it keeps
# the surrounding play parseable, which is what the benchmark needs today.
NON_SPOKEN_CHILDREN = {
	"floatingText",
	"list",
	"closer",
	"opener",
	"argument",
	"byline",
	"epigraph",
	"signed",
	"docAuthor",
	"docTitle",
	"docDate",
	"salute",
	"dateline",
}

# Captions and page furniture that carry no spoken text. <label> titles inset
# songs ("Song.") the same way <head> titles a scene. <gap>/<cb> mark damage and
# column breaks in the source witness; <q>/<quote> are quoted matter inside a
# speech, whose text the speech itself already carries.
IGNORED_CHILDREN = {
	"head",
	"label",
	"pb",
	"lb",
	"trailer",
	"milestone",
	"gap",
	"cb",
	"q",
	"quote",
	"ab",
	"note",
	"figure",
	"space",
	"bibl",
	"table",
} | NON_SPOKEN_CHILDREN


def dispatch_children(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	return parser.dispatch_children(tag)


def handle_div(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	# A <div> without @type is untyped, not malformed: some English editions use
	# one to group a dumb-show or a run of stage business. Treating it as a plain
	# section keeps its contents addressable instead of failing the whole play.
	div_type = _attr(tag, "type") or "section"
	match div_type:
		case "act":
			return _handle_act(tag, parser)
		case "scene":
			return _handle_scene(tag, parser)
		case _:
			return _handle_section(tag, parser, div_type)


def _div_number(tag: bs4.element.Tag, div_type: str) -> int | None:
	"""The act or scene number, from `@n`, else the German ordinal heading, else
	the position among sibling divs of the same type.

	TextGrid-sourced German editions carry no `@n` at all, so without the last
	two fallbacks every German act and scene would parse with an empty node
	reference and nothing could be aligned against the English text.
	"""
	own_n = _attr(tag, "n")
	if own_n.isdigit():
		return int(own_n)

	from_heading = parse_german_ordinal(tag_text(tag.find("head", recursive=False)))
	if from_heading is not None:
		return from_heading

	# Headings that name a place ("Ein Zimmer.") or a relative position
	# ("Letzter Auftritt") carry no number, but the divs are in reading order.
	position = 1
	for sibling in tag.previous_siblings:
		if isinstance(sibling, bs4.element.Tag) and sibling.name == "div" and _attr(sibling, "type") == div_type:
			position += 1
	return position


def _handle_act(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	act_number = _div_number(tag, "act")
	act_id = _attr_optional(tag, "xml:id") or f"act_{act_number}"
	display_text = ref_display_text("Act", (act_number,)) or tag_text(tag.find("head", recursive=False)) or fallback_display_text(act_id)
	result = ParseResult()
	result.add_node(DocumentNode(node_id=act_id, node_reference=(act_number, None, None), node_type=StructuralNodeType.ACT, tag="act", text=None, display_text=display_text))
	result += _consume_container_children(tag, parser, act_id, CONTAINER_CHILDREN)
	return result


def _handle_scene(tag: bs4.element.Tag, parser: DraCorTEIParser) -> ParseResult:
	parent = tag.parent if isinstance(tag.parent, bs4.element.Tag) else None
	act_number = _div_number(parent, "act") if parent is not None and parent.name == "div" and _attr(parent, "type") == "act" else None
	scene_number = _div_number(tag, "scene")
	# Scene numbers restart every act, so an unqualified `scene_1` would collide
	# across acts and silently merge two scenes into one node.
	scene_id = _attr_optional(tag, "xml:id") or (f"act_{act_number}_scene_{scene_number}" if act_number is not None else parser.next_synth_id("scene"))
	display_text = ref_display_text("Scene", (act_number, scene_number)) or tag_text(tag.find("head", recursive=False)) or fallback_display_text(scene_id)
	result = ParseResult()
	result.add_node(DocumentNode(node_id=scene_id, node_reference=(act_number, scene_number, None), node_type=StructuralNodeType.SCENE, tag="scene", text=None, display_text=display_text))
	result += _consume_container_children(tag, parser, scene_id, CONTAINER_CHILDREN)
	return result


def _handle_section(tag: bs4.element.Tag, parser: DraCorTEIParser, section_type: str) -> ParseResult:
	section_id = _attr_optional(tag, "xml:id") or parser.next_synth_id(section_type)
	result = ParseResult()
	result.add_node(DocumentNode(node_id=section_id, node_reference=(None, None, None), node_type=StructuralNodeType.SECTION, tag=section_type, text=None, display_text=tag_text(tag.find("head", recursive=False)) or section_type))
	result += _consume_container_children(tag, parser, section_id, CONTAINER_CHILDREN)
	return result


def _consume_container_children(tag: bs4.element.Tag, parser: DraCorTEIParser, parent_id: str, allowed: set[str]) -> ParseResult:
	result = ParseResult()
	child_ids: list[str] = []
	for child in tag.children:
		if not isinstance(child, bs4.element.Tag):
			continue
		if child.name in IGNORED_CHILDREN:
			continue
		if child.name not in allowed:
			raise ValueError(f"Unexpected tag <{child.name}> inside section {parent_id}")
		child_result = parser.dispatch(child)
		if child_result.nodes:
			child_ids.append(child_result.nodes[0].node_id)
		result += child_result
	for i in range(1, len(child_ids)):
		result.add_edge(child_ids[i - 1], StructuralEdgeType.NEXT, child_ids[i])
	for child_id in child_ids:
		result.add_edge(parent_id, StructuralEdgeType.CONTAINS, child_id)
	return result
