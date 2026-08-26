from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import bs4
from networkx import MultiDiGraph

from critics_eye.core.models import StructuralEdgeType

from ._models import ElementHandler, ParseResult

logger = logging.getLogger(__name__)


class TEIParser:
	"""Registry-based TEI parser with visitor dispatch."""

	def __init__(
		self,
		body_handlers: dict[str, ElementHandler],
		header_handlers: dict[str, ElementHandler],
	) -> None:
		self._body_handlers = body_handlers
		self._header_handlers = header_handlers
		self._active_handlers: dict[str, ElementHandler] = body_handlers

	def dispatch(self, tag: bs4.element.Tag) -> ParseResult:
		"""Dispatch a single tag to its registered handler."""
		handler = self._active_handlers.get(tag.name)
		if handler is None:
			raise ValueError(
				f"No handler registered for <{tag.name}> "
				f"(attrs={dict(tag.attrs)}, text={tag.text.strip()[:60]!r}...)"
			)
		return handler(tag, self)

	def dispatch_children(self, parent: bs4.element.Tag) -> ParseResult:
		"""Dispatch all direct Tag children of parent, accumulating results."""
		result = ParseResult()
		for child in parent.children:
			if isinstance(child, bs4.element.Tag):
				result += self.dispatch(child)
		return result

	def parse(self, path: Path, show_progress: bool = True) -> MultiDiGraph:
		"""Parse a TEI XML file into a NetworkX MultiDiGraph."""
		from rich.progress import (
			Progress,
			SpinnerColumn,
			TextColumn,
			BarColumn,
			MofNCompleteColumn,
		)

		soup = bs4.BeautifulSoup(path.read_text(encoding="utf-8"), "xml")
		root = soup.find("TEI")
		assert root, f"No <TEI> root element in {path}"

		play_name = path.stem

		# Count div1s for progress bar granularity
		text_el = root.find("text")
		assert text_el, f"No <text> in {path}"
		body = text_el.find("body")
		assert body, f"No <body> in {path}"
		div1_count = len(body.find_all("div1", recursive=False))

		with Progress(
			SpinnerColumn(),
			TextColumn("[bold blue]{task.description}"),
			BarColumn(),
			MofNCompleteColumn(),
			disable=not show_progress,
			transient=True,
		) as progress:
			# total = 1 (header) + div1_count (acts) + 1 (assemble)
			task = progress.add_task(f"Parsing {play_name}", total=div1_count + 2)

			# --- Header ---
			tei_header = root.find("teiHeader")
			assert tei_header, f"No <teiHeader> in {path}"
			# A cast list is conventional but not required: a dozen Early Modern
			# English plays in DraCor have no <particDesc>. Their speeches still
			# name their speakers, so the play parses -- it just contributes no
			# character nodes rather than being dropped entirely.
			partic_desc = tei_header.find("particDesc")

			self._active_handlers = self._header_handlers
			header_result = (
				self.dispatch_children(partic_desc) if partic_desc else ParseResult()
			)
			progress.update(task, advance=1, description=f"{play_name}: characters")

			# --- Body (per-div1 progress) ---
			self._active_handlers = self._body_handlers
			body_result = self._parse_body_with_progress(
				body, play_name, progress, task
			)

			# --- Assemble ---
			result = ParseResult()
			result += header_result
			result += body_result

			character_ids = [n.node_id for n in header_result.nodes]
			for char_id in character_ids:
				result.add_edge(char_id, StructuralEdgeType.APPEARS_IN, play_name)

			progress.update(task, advance=1, description=f"{play_name}: done")

		return self._build_graph(result, play_name)

	def _parse_body_with_progress(
		self,
		body: bs4.element.Tag,
		play_name: str,
		progress: Any,
		task_id: Any,
	) -> ParseResult:
		"""Parse <body> with per-div1 progress updates."""
		from critics_eye.core.models import DocumentNode, StructuralNodeType

		result = ParseResult()

		play_node = DocumentNode(
			node_id=play_name,
			node_reference=(None, None, None),
			node_type=StructuralNodeType.PLAY,
			tag="play",
			text=None,
			display_text=play_name,
		)
		result.add_node(play_node)

		div1s = body.find_all("div1", recursive=False)
		child_ids: list[str] = []

		for div1 in div1s:
			div1_type = div1.get("type", "act")
			div1_n = div1.get("n", "?")
			progress.update(
				task_id, advance=1, description=f"{play_name}: {div1_type} {div1_n}"
			)

			child_result = self.dispatch(div1)
			if child_result.nodes:
				child_ids.append(child_result.nodes[0].node_id)
			result += child_result

		# Sequential edges between acts/prologues
		for i in range(1, len(child_ids)):
			result.add_edge(child_ids[i - 1], StructuralEdgeType.NEXT, child_ids[i])

		# Cross-act line→line NEXT: rebuild full play-level line chain.
		# Act-level handlers already built intra-act chains; just add cross-act links.
		line_id_set = set(result.line_ids)
		result.edges = [
			(s, et, t)
			for s, et, t in result.edges
			if not (
				et == StructuralEdgeType.NEXT and s in line_id_set and t in line_id_set
			)
		]
		for i in range(1, len(result.line_ids)):
			result.add_edge(
				result.line_ids[i - 1], StructuralEdgeType.NEXT, result.line_ids[i]
			)

		# Parent edges: play contains acts
		for child_id in child_ids:
			result.add_edge(play_name, StructuralEdgeType.CONTAINS, child_id)

		return result

	@staticmethod
	def _build_graph(result: ParseResult, play_name: str) -> MultiDiGraph:
		"""Convert ParseResult into a NetworkX MultiDiGraph."""
		graph = MultiDiGraph(graph_id=play_name)
		known_node_ids: set[str] = set()

		for node in result.nodes:
			graph.add_node(node.node_id, **node.model_dump())
			known_node_ids.add(node.node_id)

		skipped_edges = 0
		for src, edge_type, dst in result.edges:
			if src not in known_node_ids or dst not in known_node_ids:
				skipped_edges += 1
				logger.warning(
					"Skipping dangling edge %r --%s--> %r",
					src,
					edge_type,
					dst,
				)
				continue
			graph.add_edge(src, dst, key=None, edge_type=edge_type)

		logger.info(
			f"Graph built: {graph.number_of_nodes()} nodes, "
			f"{graph.number_of_edges()} edges"
		)
		if skipped_edges:
			logger.warning("Skipped %s dangling edge(s) during graph build.", skipped_edges)
		return graph
