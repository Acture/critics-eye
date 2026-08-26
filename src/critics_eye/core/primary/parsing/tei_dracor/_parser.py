from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import bs4
from networkx import MultiDiGraph

from critics_eye.core.models import DocumentNode, StructuralEdgeType, StructuralNodeType
from critics_eye.core.primary.parsing.tei._parser import TEIParser

from ._models import ElementHandler, ParseResult

logger = logging.getLogger(__name__)


class DraCorTEIParser(TEIParser):
	def __init__(
		self,
		body_handlers: dict[str, ElementHandler],
		header_handlers: dict[str, ElementHandler],
	) -> None:
		super().__init__(body_handlers=body_handlers, header_handlers=header_handlers)
		self._synth_counters: dict[str, int] = defaultdict(int)

	def next_synth_id(self, prefix: str) -> str:
		self._synth_counters[prefix] += 1
		return f"{prefix}_{self._synth_counters[prefix]}"

	def parse(self, path: Path, show_progress: bool = True) -> MultiDiGraph:
		self._synth_counters.clear()
		return super().parse(path, show_progress=show_progress)

	def _parse_body_with_progress(
		self,
		body: bs4.element.Tag,
		play_name: str,
		progress: Any,
		task_id: Any,
	) -> ParseResult:
		result = ParseResult()
		result.add_node(
			DocumentNode(
				node_id=play_name,
				node_reference=(None, None, None),
				node_type=StructuralNodeType.PLAY,
				tag="play",
				text=None,
				display_text=play_name,
			)
		)

		divs = [child for child in body.children if isinstance(child, bs4.element.Tag) and child.name == "div"]
		child_ids: list[str] = []
		for div in divs:
			progress.update(task_id, advance=1, description=f"{play_name}: {div.get('type', 'div')} {div.get('n', '?')}")
			child_result = self.dispatch(div)
			if child_result.nodes:
				child_ids.append(child_result.nodes[0].node_id)
			result += child_result

		for i in range(1, len(child_ids)):
			result.add_edge(child_ids[i - 1], StructuralEdgeType.NEXT, child_ids[i])

		line_id_set = set(result.line_ids)
		result.edges = [
			(s, et, t)
			for s, et, t in result.edges
			if not (et == StructuralEdgeType.NEXT and s in line_id_set and t in line_id_set)
		]
		for i in range(1, len(result.line_ids)):
			result.add_edge(result.line_ids[i - 1], StructuralEdgeType.NEXT, result.line_ids[i])

		for child_id in child_ids:
			result.add_edge(play_name, StructuralEdgeType.CONTAINS, child_id)
		return result

	def parse_counts(self, path: Path) -> tuple[int, int]:
		graph = self.parse(path, show_progress=False)
		return graph.number_of_nodes(), graph.number_of_edges()
