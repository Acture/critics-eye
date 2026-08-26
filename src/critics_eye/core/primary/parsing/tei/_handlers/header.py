from __future__ import annotations

from typing import TYPE_CHECKING, cast

import bs4

from critics_eye.core.models import DocumentNode, KeyNodeType, SocialEdgeType

from .._labels import fallback_display_text, first_tag_text, tag_text
from .._models import ParseResult

if TYPE_CHECKING:
	from .._parser import TEIParser


def _attr_optional(tag: bs4.element.Tag, key: str) -> str | None:
	value = tag.get(key)
	if value is None:
		return None
	return cast(str, value)


def _iter_ref_targets(tag: bs4.element.Tag) -> list[str]:
	ref = _attr_optional(tag, "ref")
	if not ref:
		return []
	return [part.lstrip("#") for part in ref.split() if part.strip()]


def handle_list_person(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <listPerson> — dispatch person/personGrp children."""
	return parser.dispatch_children(tag)


def handle_person(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <person> — create a CHARACTER node."""
	xml_id = _attr_optional(tag, "xml:id") or _attr_optional(tag, "sameAs")
	if not xml_id:
		# Some plays have empty <person/> tags (e.g., TNK). Skip them.
		return ParseResult()

	# Collect descriptive text from child elements
	text_parts: list[str] = []
	for child in tag.children:
		if not isinstance(child, bs4.element.Tag):
			continue
		child_text = child.get_text(separator=" ", strip=True)
		if child_text:
			text_parts.append(child_text)
	description = " | ".join(text_parts) or None
	display_text = (
		first_tag_text(tag, ("persName", "name"), recursive=True)
		or tag_text(tag.find("p", recursive=False))
		or fallback_display_text(xml_id)
	)

	char_node = DocumentNode(
		node_id=xml_id,
		node_reference=(None, None, None),
		node_type=KeyNodeType.CHARACTER,
		tag="person",
		text=description,
		display_text=display_text,
	)

	result = ParseResult()
	result.add_node(char_node)

	# <rs> children → RELATED_TO edges via ref attribute
	for rs in tag.find_all("rs", recursive=True):
		for target_id in _iter_ref_targets(rs):
			result.add_edge(xml_id, SocialEdgeType.RELATED_TO, target_id)

	return result


def handle_person_group(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <personGrp> — create a GROUP node."""
	xml_id = _attr_optional(tag, "xml:id")
	if not xml_id:
		# Synthesize from group text content
		text = tag.get_text(strip=True)[:30].replace(" ", "_")
		xml_id = f"group_{text}" if text else f"group_{id(tag)}"

	display_text = (
		first_tag_text(tag, ("persName", "name", "p"), recursive=True)
		or fallback_display_text(xml_id)
	)

	group_node = DocumentNode(
		node_id=xml_id,
		node_reference=(None, None, None),
		node_type=KeyNodeType.GROUP,
		tag="personGrp",
		text=None,
		display_text=display_text,
	)

	result = ParseResult()
	result.add_node(group_node)

	# <rs> children → MEMBER_OF edges for referenced persons
	for rs in tag.find_all("rs", recursive=True):
		for target_id in _iter_ref_targets(rs):
			result.add_edge(target_id, SocialEdgeType.MEMBER_OF, xml_id)

	return result


def handle_rs(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <rs> — consumed by parent handlers (person/personGrp); skip here."""
	return ParseResult()


def extract_text(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle persName, sex, state, etc. — consumed by handle_person; no nodes produced."""
	return ParseResult()
