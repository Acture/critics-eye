from __future__ import annotations

from typing import TYPE_CHECKING

import bs4

from critics_eye.core.models import DocumentNode, StructuralEdgeType, StructuralNodeType

from .._labels import excerpt_display_text, fallback_display_text, ref_display_text
from .._models import ParseResult
from .structural import _attr

if TYPE_CHECKING:
	from .._parser import TEIParser


def _parse_ref(ref_str: str) -> tuple[int | None, ...]:
	"""Parse a reference like '1.2.34' or 'IND.1.5' into a tuple, using None for non-numeric parts."""
	return tuple(int(p) if p.isdigit() else None for p in ref_str.split("."))


def handle_speech(tag: bs4.element.Tag, parser: TEIParser) -> ParseResult:
	"""Handle a <sp> speech element.

	Two-phase approach:
	  1. dispatch_children(ab) → collects all atoms (w/c/pc) recursively
	  2. iterate milestones manually → reconstruct LINE nodes from collected atoms
	"""
	result = ParseResult()

	# --- Speech ID (with fallback for the one Hamlet edge case) ---
	speech_id = _attr(tag, "xml:id") or None
	if not speech_id:
		first_ms = tag.find("milestone", attrs={"unit": "ftln"})
		assert first_ms is not None, f"<sp> missing xml:id and no ftln milestone: {tag}"
		assert isinstance(first_ms, bs4.element.Tag)
		n_val = _attr(first_ms, "n")
		assert n_val, f"<sp> fallback milestone has no 'n' attribute: {tag}"
		speech_id = "sp_" + n_val.replace(".", "_")

	# --- Content container: <ab> or <p> (epilogues use <p>) ---
	content_container = tag.find("ab") or tag.find("p")

	# --- Speech node reference: from first non-SD atom ---
	ref: tuple[int | None, ...] = (None, None, None)
	if content_container:
		for candidate in content_container.find_all(["w", "c"]):
			assert isinstance(candidate, bs4.element.Tag)
			n_val = _attr(candidate, "n")
			if n_val and not n_val.startswith("SD"):
				ref = _parse_ref(n_val)
				break

	# --- Create speech SECTION node ---
	speech_node = DocumentNode(
		node_id=speech_id,
		node_reference=ref,
		node_type=StructuralNodeType.SECTION,
		tag="speech",
		text=None,
		display_text=ref_display_text("Speech", ref) or fallback_display_text(speech_id),
	)
	result.add_node(speech_node)

	# --- SPOKEN_BY edges from who attribute ---
	for raw in _attr(tag, "who").split():
		speaker_id = raw.lstrip("#")
		if speaker_id:
			result.add_edge(speech_id, StructuralEdgeType.SPOKEN_BY, speaker_id)

	if not content_container:
		# No <ab> or <p> — speech has only stage directions (e.g., "sings")
		# Just dispatch the sp's children directly for any stages inside
		result += parser.dispatch_children(tag)
		return result

	# --- Phase 1: dispatch all children of content container to collect atoms ---
	children_result = parser.dispatch_children(content_container)
	result += children_result

	# --- Phase 2: collect milestones and reconstruct LINE nodes ---
	milestones: list[tuple[str, tuple[int | None, ...], list[str]]] = []

	for ms in content_container.find_all("milestone"):
		assert isinstance(ms, bs4.element.Tag)
		if _attr(ms, "unit") == "page":
			continue
		ms_id = _attr(ms, "xml:id")
		assert ms_id, f"Milestone missing xml:id in speech {speech_id!r}"
		corresp = _attr(ms, "corresp")
		if not corresp:
			continue
		atom_ids = [c.lstrip("#") for c in corresp.split() if c.lstrip("#")]
		n_attr = _attr(ms, "n")
		assert (
			n_attr
		), f"Milestone {ms_id!r} missing 'n' attribute in speech {speech_id!r}"
		ms_ref = _parse_ref(n_attr)
		milestones.append((ms_id, ms_ref, atom_ids))

	line_ids: list[str] = []
	for ms_id, ms_ref, atom_ids in milestones:
		text = "".join(
			children_result.atoms[aid]
			for aid in atom_ids
			if aid in children_result.atoms
		)
		line_node = DocumentNode(
			node_id=ms_id,
			node_reference=ms_ref,
			node_type=StructuralNodeType.LINE,
			text=text.strip(),
			display_text=excerpt_display_text(text),
			tag="line",
		)
		result.add_node(line_node)
		result.add_edge(speech_id, StructuralEdgeType.CONTAINS, ms_id)
		line_ids.append(ms_id)

	# NEXT edges between sequential lines (within this speech)
	for i in range(1, len(line_ids)):
		result.add_edge(line_ids[i - 1], StructuralEdgeType.NEXT, line_ids[i])

	# Expose line ordering for cross-speech next edges (built by handle_scene)
	result.line_ids = line_ids

	return result
