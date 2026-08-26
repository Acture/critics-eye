"""Variant B: Convert hierarchical HeteroData to line-only closure-edge graph.

Instead of the 5-level hierarchy (Play→Act→Scene→Section→Line), this collapses
structural proximity into typed line↔line edges:
- (line, next, line) — sequential flow (preserved from Variant A)
- (line, same_section, line) — lines sharing a section (speech unit)
- (line, same_scene, line) — lines sharing a scene
- (line, same_act, line) — lines sharing an act
- (line, same_speaker, line) — lines spoken by the same character
"""

import logging

import torch
from torch_geometric.data import HeteroData

from critics_eye.core.models import StructuralEdgeType
from critics_eye.utils.data_prep import infer_line_act_ids


def _build_cooccurrence_edges(
	group_ids: torch.Tensor,
	max_group_size: int = 500,
) -> torch.Tensor:
	"""Build undirected edges between all nodes sharing the same group ID.

	Args:
		group_ids: [N] tensor mapping each node to a group
		max_group_size: skip groups larger than this (performance guard)

	Returns:
		edge_index: [2, E] undirected edges
	"""
	srcs: list[torch.Tensor] = []
	dsts: list[torch.Tensor] = []

	for gid in group_ids.unique():
		members = torch.where(group_ids == gid)[0]
		if len(members) < 2:
			continue
		if len(members) > max_group_size:
			logging.warning(
				f"Group {gid.item()} has {len(members)} members, skipping (> {max_group_size})"
			)
			continue

		# All-pairs within group (excluding self-loops)
		grid = torch.cartesian_prod(members, members)
		mask = grid[:, 0] != grid[:, 1]
		pairs = grid[mask]
		srcs.append(pairs[:, 0])
		dsts.append(pairs[:, 1])

	if not srcs:
		return torch.zeros((2, 0), dtype=torch.long)

	return torch.stack([torch.cat(srcs), torch.cat(dsts)])


def build_closure_graph(hierarchical_data: HeteroData) -> HeteroData:
	"""Convert a Variant A hierarchical graph to Variant B line-only closure graph.

	Preserves line node features and 'next' edges. Adds closure edges encoding
	structural co-occurrence at different granularities.

	Args:
		hierarchical_data: full hierarchical HeteroData (Variant A)

	Returns:
		New HeteroData with only 'line' nodes and closure edge types
	"""
	data = hierarchical_data
	num_lines = data["line"].num_nodes

	closure = HeteroData()
	closure["line"].x = data["line"].x.clone()
	closure["line"].num_nodes = num_lines

	# Preserve text metadata if present
	if hasattr(data["line"], "text"):
		closure["line"].text = data["line"].text

	# 1. Preserve sequential edges
	if ("line", StructuralEdgeType.NEXT, "line") in data.edge_types:
		closure["line", "next", "line"].edge_index = data[
			("line", StructuralEdgeType.NEXT, "line")
		].edge_index.clone()
		logging.info(
			f"next edges: {closure['line', 'next', 'line'].edge_index.size(1)}"
		)

	# 2. Same-section edges (lines in same speech unit)
	section_of_line = torch.zeros(num_lines, dtype=torch.long)
	if ("section", StructuralEdgeType.CONTAINS, "line") in data.edge_types:
		ei = data[("section", StructuralEdgeType.CONTAINS, "line")].edge_index
		section_of_line[ei[1]] = ei[0]

		edge_index = _build_cooccurrence_edges(section_of_line)
		closure["line", "same_section", "line"].edge_index = edge_index
		logging.info(f"same_section edges: {edge_index.size(1)}")

	# 3. Same-scene edges
	scene_of_section = torch.zeros(data["section"].num_nodes, dtype=torch.long)
	if ("scene", StructuralEdgeType.CONTAINS, "section") in data.edge_types:
		ei = data[("scene", StructuralEdgeType.CONTAINS, "section")].edge_index
		scene_of_section[ei[1]] = ei[0]

	line_scene_ids = scene_of_section[section_of_line]
	edge_index = _build_cooccurrence_edges(line_scene_ids)
	closure["line", "same_scene", "line"].edge_index = edge_index
	logging.info(f"same_scene edges: {edge_index.size(1)}")

	# 4. Same-act edges
	line_act_ids = infer_line_act_ids(data)
	edge_index = _build_cooccurrence_edges(line_act_ids)
	closure["line", "same_act", "line"].edge_index = edge_index
	logging.info(f"same_act edges: {edge_index.size(1)}")

	# 5. Same-speaker edges (if character info available)
	sec2char_type = ("section", StructuralEdgeType.SPOKEN_BY, "character")
	sec2line_type = ("section", StructuralEdgeType.CONTAINS, "line")

	if sec2char_type in data.edge_types and sec2line_type in data.edge_types:
		sec2char = data[sec2char_type].edge_index

		# section → character mapping (multi-speaker sections get -1)
		num_sections = data["section"].num_nodes
		char_of_section = torch.full((num_sections,), -1, dtype=torch.long)
		for s in sec2char[0].unique():
			chars = sec2char[1][sec2char[0] == s].unique()
			if chars.numel() == 1:
				char_of_section[s] = chars[0]

		# line → character via section
		line_char_ids = char_of_section[section_of_line]
		# Only include lines with known single speaker
		valid_mask = line_char_ids >= 0

		# Build edges only among valid lines
		valid_indices = torch.where(valid_mask)[0]
		valid_char_ids = line_char_ids[valid_indices]

		# Remap to build cooccurrence, then map back
		edge_index_local = _build_cooccurrence_edges(valid_char_ids)
		if edge_index_local.size(1) > 0:
			edge_index_global = valid_indices[edge_index_local]
			closure["line", "same_speaker", "line"].edge_index = edge_index_global
			logging.info(f"same_speaker edges: {edge_index_global.size(1)}")
		else:
			closure["line", "same_speaker", "line"].edge_index = torch.zeros(
				(2, 0), dtype=torch.long
			)
			logging.info("same_speaker edges: 0")

	# Copy theme data if present (for motif training)
	if "theme" in data.node_types:
		closure["theme"].x = data["theme"].x.clone()
		closure["theme"].num_nodes = data["theme"].num_nodes
		if hasattr(data["theme"], "text"):
			closure["theme"].text = data["theme"].text
		if hasattr(data["theme"], "labels"):
			closure["theme"].labels = data["theme"].labels

		# Copy line→theme edges
		for et in data.edge_types:
			if et[0] == "line" and et[2] == "theme":
				closure[et].edge_index = data[et].edge_index.clone()

	logging.info(
		f"Closure graph: {num_lines} lines, "
		f"{len(closure.edge_types)} edge types, "
		f"{sum(closure[et].edge_index.size(1) for et in closure.edge_types)} total edges"
	)

	return closure
