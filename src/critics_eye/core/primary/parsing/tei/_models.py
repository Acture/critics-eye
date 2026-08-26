from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import bs4

from critics_eye.core.models import DocumentNode, EdgeType

if TYPE_CHECKING:
	from ._parser import TEIParser


@dataclass
class ParseResult:
	"""Accumulated result of parsing a TEI subtree."""

	nodes: list[DocumentNode] = field(default_factory=list)
	edges: list[tuple[str, EdgeType, str]] = field(default_factory=list)
	atoms: dict[str, str] = field(default_factory=dict)
	line_ids: list[str] = field(default_factory=list)
	"""Ordered line IDs produced by this subtree, used to build cross-speech next edges."""

	def add_node(self, node: DocumentNode) -> None:
		self.nodes.append(node)

	def add_edge(self, source: str, edge_type: EdgeType, target: str) -> None:
		self.edges.append((source, edge_type, target))

	def __iadd__(self, other: ParseResult) -> ParseResult:
		self.nodes.extend(other.nodes)
		self.edges.extend(other.edges)
		self.atoms.update(other.atoms)
		self.line_ids.extend(other.line_ids)
		return self


class ElementHandler(Protocol):
	def __call__(self, tag: bs4.element.Tag, parser: TEIParser) -> ParseResult: ...
