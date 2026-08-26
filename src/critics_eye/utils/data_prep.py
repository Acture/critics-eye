"""Shared data preparation utilities for HeteroData graphs.

Consolidates duplicated logic from train/evaluate/predict modules:
- build_x_dict: GNN input dictionary construction
- collect_line_theme_edges: gather ALL line→theme edge types
- infer_line_act_ids: line → act ID chain inference
- split_edges_by_act: LOAO edge splitting
"""

import logging
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from critics_eye.core.models import StructuralEdgeType


def collect_line_theme_edges(
	data: HeteroData,
	membership_relation: str | None = None,
) -> Tensor:
	"""Collect line→theme supervision edges.

	By default this preserves the historical behavior and concatenates every
	line→theme relation. ``membership_relation`` opt-in restricts supervision
	without changing the graph used for message passing.
	"""
	line_theme_edges: list[Tensor] = []
	for et in data.edge_types:
		relation = str(getattr(et[1], "value", et[1]))
		if (
			et[0] == "line"
			and et[2] == "theme"
			and (membership_relation is None or relation == membership_relation)
		):
			line_theme_edges.append(data[et].edge_index)
	if not line_theme_edges:
		scope = (
			"all relations"
			if membership_relation is None
			else f"relation {membership_relation!r}"
		)
		raise ValueError(f"No line→theme edges found for {scope}")
	edge_index = torch.cat(line_theme_edges, dim=1)
	logging.info(
		f"Collected {edge_index.shape[1]} line→theme edges "
		f"from {len(line_theme_edges)} edge type(s)"
		+ (
			""
			if membership_relation is None
			else f" using membership_relation={membership_relation!r}"
		)
	)
	return edge_index


def build_x_dict(
	data: HeteroData,
	device: torch.device,
) -> dict[str, Tensor]:
	"""Build GNN input feature dictionary from HeteroData.

	For node types with features (.x), moves them to device.
	For node types without features, creates zero-padded placeholders (dim=1)
	so that GeneralGNN can generate learnable embeddings internally.
	"""
	x_dict: dict[str, Tensor] = {}
	for node_type in data.node_types:
		node_store = data[node_type]
		if "x" in node_store and node_store.x is not None:
			x_dict[node_type] = node_store.x.to(device)
		else:
			num_nodes = data[node_type].num_nodes
			if num_nodes is None:
				logging.warning(
					f"Node type '{node_type}' has no num_nodes info. Skipping."
				)
				continue
			if num_nodes == 0:
				continue
			x_dict[node_type] = torch.zeros((num_nodes, 1), device=device)
	return x_dict


def infer_line_act_ids(data: HeteroData) -> Tensor:
	"""Infer act ID for each line node via edge chain traversal.

	If data['line'].act_id exists, returns it directly.
	Otherwise traces: line → section → scene → act using structural edges.

	Returns:
		Tensor of shape [num_lines] with 0-indexed act IDs.
	"""
	if hasattr(data["line"], "act_id"):
		return data["line"].act_id

	# Line → Section
	section_of_line = torch.zeros(data["line"].num_nodes, dtype=torch.long)
	if ("section", StructuralEdgeType.CONTAINS, "line") in data.edge_types:
		edge_l2s = data[("section", StructuralEdgeType.CONTAINS, "line")].edge_index
		section_of_line[edge_l2s[1]] = edge_l2s[0]
	else:
		logging.warning("Edge (section -> line) not found!")

	# Section → Scene
	scene_of_section = torch.zeros(data["section"].num_nodes, dtype=torch.long)
	if ("scene", StructuralEdgeType.CONTAINS, "section") in data.edge_types:
		edge_s2s = data[("scene", StructuralEdgeType.CONTAINS, "section")].edge_index
		scene_of_section[edge_s2s[1]] = edge_s2s[0]
	else:
		logging.warning("Edge (scene -> section) not found!")

	# Scene → Act
	act_of_scene = torch.zeros(data["scene"].num_nodes, dtype=torch.long)
	if ("act", StructuralEdgeType.CONTAINS, "scene") in data.edge_types:
		edge_s2a = data[("act", StructuralEdgeType.CONTAINS, "scene")].edge_index
		act_of_scene[edge_s2a[1]] = edge_s2a[0]
	else:
		logging.warning("Edge (act -> scene) not found!")

	# Chain: line → section → scene → act
	line_scene_ids = scene_of_section[section_of_line]
	line_act_ids = act_of_scene[line_scene_ids]

	logging.info(f"Inferred Act IDs shape: {line_act_ids.shape}")
	logging.info(f"Unique Acts found: {line_act_ids.unique().tolist()}")
	num_acts = int(line_act_ids.max().item()) + 1
	logging.info(f"Lines per Act: {torch.bincount(line_act_ids, minlength=num_acts)}")

	return line_act_ids


def split_edges_by_act(
	src: Tensor,
	dst: Tensor,
	line_act_ids: Tensor,
	target_act: int,
	device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
	"""Split edges into train/val sets based on LOAO (Leave-One-Act-Out).

	Args:
		src: source node indices (line IDs)
		dst: destination node indices (theme IDs)
		line_act_ids: act ID per line (0-indexed)
		target_act: 1-indexed act ID to leave out
		device: target device for output tensors

	Returns:
		(train_src, train_dst, val_src, val_dst) — all on device
	"""
	target_act_idx = target_act - 1

	src_cpu = src.cpu()
	line_act_ids_cpu = line_act_ids.cpu()

	is_test_line = line_act_ids_cpu == target_act_idx
	test_edge_mask = is_test_line[src_cpu]
	train_edge_mask = ~test_edge_mask

	num_train = train_edge_mask.sum().item()
	num_val = test_edge_mask.sum().item()

	logging.info(f"LOAO split: Train={num_train}, Val={num_val} (Act {target_act})")

	if num_val == 0:
		logging.warning(
			f"No edges found for Act {target_act}! Validation will return 0.0."
		)

	train_src = src[train_edge_mask].to(device)
	train_dst = dst[train_edge_mask].to(device)
	val_src = src[test_edge_mask].to(device)
	val_dst = dst[test_edge_mask].to(device)

	return train_src, train_dst, val_src, val_dst


def filter_edges_by_act(
	src_gt: Tensor,
	dst_gt: Tensor,
	line_act_ids: Tensor,
	target_act: int,
) -> tuple[Tensor, Tensor]:
	"""Filter edges to keep only those where the source line belongs to target_act.

	Used in evaluation to restrict metrics to a specific act.

	Args:
		target_act: 1-indexed act ID to keep
	"""
	target_act_idx = target_act - 1
	src_cpu = src_gt.cpu()
	line_act_ids_cpu = line_act_ids.cpu()
	mask = (line_act_ids_cpu[src_cpu] == target_act_idx).to(src_gt.device)
	return src_gt[mask], dst_gt[mask]


def build_existing_line_theme_mask(
	data: HeteroData,
	num_lines: int,
	num_themes: int,
) -> Tensor:
	"""Build boolean mask of existing line→theme edges for masking predictions."""
	mask = torch.zeros((num_lines, num_themes), dtype=torch.bool)
	for et in data.edge_types:
		if et[0] == "line" and et[2] == "theme":
			ei = data[et].edge_index
			mask[ei[0], ei[1]] = True
	return mask
