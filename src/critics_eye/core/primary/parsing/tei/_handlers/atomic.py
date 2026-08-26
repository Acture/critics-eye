from __future__ import annotations

from typing import TYPE_CHECKING

import bs4

from .._models import ParseResult

if TYPE_CHECKING:
	from .._parser import TEIParser


def handle_atom(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <w>, <c>, <pc> atomic token tags."""
	xml_id: str | None = tag.get("xml:id")
	assert xml_id, f"<{tag.name}> missing xml:id: {tag}"

	text: str = tag.get_text()

	result = ParseResult()
	result.atoms[xml_id] = text
	return result


def skip(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Return empty ParseResult, discarding the tag."""
	return ParseResult()


def dispatch_children(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Transparent wrapper — dispatch all children of container tags like <q>, <ab>, <hi>."""
	return parser.dispatch_children(tag)


def handle_speaker(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle <speaker> tag. Character info is authoritative from sp[@who]; skip here."""
	return ParseResult()
