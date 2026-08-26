"""Structural counterfactual perturbations for HeteroData graphs.

Each perturbation takes a ``HeteroData`` and a seed, returns a NEW
``HeteroData`` (the input is never mutated). Randomness uses an explicit
``torch.Generator`` so calls are reproducible.

Three perturbations are exposed:

* :func:`perturb_scenes` (P1) — within each act, randomly reassign sections to
  scenes. The line→line ``next`` chain is preserved.
* :func:`perturb_speakers` (P2) — permute the target column of every
  ``section → {character, group}`` ``spoken_by`` edge.
* :func:`perturb_random_tree` (P4) — keep the act/scene/section hierarchy
  intact but randomly permute the ``section → line`` leaf assignment so every
  line is attached to a (uniformly) random section.

These functions are intentionally schema-agnostic: edge types are located by
their ``(src, dst)`` node-type pair rather than by relation-name string, so
they work whether the artifact uses the legacy ``in_section`` / ``in_act`` or
the canonical ``contains`` relation labels.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from enum import StrEnum

import torch
from torch_geometric.data import HeteroData

EdgeKey = tuple[str, str, str]


class PerturbKind(StrEnum):
	"""Identifiers accepted by the ``ce-cli primary perturb`` command."""

	SCENES = "scenes"
	SPEAKERS = "speakers"
	TREE = "tree"


def _find_edge_type(data: HeteroData, src: str, dst: str) -> EdgeKey:
	"""Locate the unique edge type with the given source and destination
	node types. Raises ``KeyError`` if missing or non-unique."""
	matches = [et for et in data.edge_types if et[0] == src and et[2] == dst]
	if not matches:
		raise KeyError(
			f"No edge type with src={src!r}, dst={dst!r}; "
			f"available={list(data.edge_types)}"
		)
	if len(matches) > 1:
		raise KeyError(f"Multiple edge types with src={src!r}, dst={dst!r}: {matches}")
	return matches[0]


def _try_find_edge_type(data: HeteroData, src: str, dst: str) -> EdgeKey | None:
	matches = [et for et in data.edge_types if et[0] == src and et[2] == dst]
	if len(matches) == 1:
		return matches[0]
	return None


def _make_generator(seed: int) -> torch.Generator:
	g = torch.Generator(device="cpu")
	g.manual_seed(int(seed))
	return g


def perturb_scenes(data: HeteroData, seed: int) -> HeteroData:
	"""P1 — within each act, randomly permute scene→section assignment.

	Preserves the total ``scene→section`` edge count and the multiset of
	scenes-per-act. Scrambles which sections (and transitively which lines)
	belong to which scene. Lines stay in the same act because the section
	tree is untouched. The ``line→line`` ``next`` chain is preserved.

	Requires the input graph to expose both ``(act, *, scene)`` and
	``(scene, *, section)`` edges.
	"""
	out = copy.deepcopy(data)
	g = _make_generator(seed)

	act_scene_et = _find_edge_type(out, "act", "scene")
	scene_sec_et = _find_edge_type(out, "scene", "section")

	num_scene = out["scene"].num_nodes
	num_act = out["act"].num_nodes

	# Build a scene → act lookup. Some artifacts share scene nodes across
	# acts (e.g. when scenes are deduplicated by ordinal name); in that case
	# we attach each scene to the first act it appears under.
	act_of_scene = torch.full((num_scene,), -1, dtype=torch.long)
	a2s = out[act_scene_et].edge_index
	for a, s in zip(a2s[0].tolist(), a2s[1].tolist()):
		if act_of_scene[s] == -1:
			act_of_scene[s] = a

	ss = out[scene_sec_et].edge_index
	scenes = ss[0].clone()
	sections = ss[1].clone()
	new_scenes = scenes.clone()

	for a in range(num_act):
		scenes_in_act = (act_of_scene == a).nonzero(as_tuple=False).flatten()
		if scenes_in_act.numel() == 0:
			continue
		# Edge positions whose scene endpoint is one of the act's scenes.
		mask = torch.zeros(scenes.shape[0], dtype=torch.bool)
		for s in scenes_in_act.tolist():
			mask |= scenes == s
		positions = mask.nonzero(as_tuple=False).flatten()
		if positions.numel() == 0:
			continue
		# Permuting the scene-column over these positions preserves the
		# per-scene in-degree multiset within the act.
		scene_col = scenes[positions]
		perm = torch.randperm(scene_col.shape[0], generator=g)
		new_scenes[positions] = scene_col[perm]

	out[scene_sec_et].edge_index = torch.stack([new_scenes, sections])
	return out


def perturb_speakers(data: HeteroData, seed: int) -> HeteroData:
	"""P2 — permute the target column of section→character / section→group.

	Each section retains its text, sub-section structure and motif edges;
	only the speaker assignment is randomized. The per-speaker degree
	distribution is preserved exactly because we permute targets in place.
	"""
	out = copy.deepcopy(data)
	g = _make_generator(seed)

	touched = False
	for dst in ("character", "group"):
		et = _try_find_edge_type(out, "section", dst)
		if et is None:
			continue
		ei = out[et].edge_index
		if ei.shape[1] == 0:
			continue
		perm = torch.randperm(ei.shape[1], generator=g)
		out[et].edge_index = torch.stack([ei[0], ei[1][perm]])
		touched = True

	if not touched:
		raise KeyError(
			"perturb_speakers requires at least one (section, *, character) "
			f"or (section, *, group) edge; available={list(out.edge_types)}"
		)

	return out


def perturb_random_tree(data: HeteroData, seed: int) -> HeteroData:
	"""P4 — randomize the line→section leaf assignment.

	Implementation note (deviation from the task brief): the brief calls for
	rebuilding the act/scene/section hierarchy from a fresh random k-ary
	tree of identical depth/internal-node count. The authoritative design
	doc (``paper/notes/v17_strength_proposals/counterfactual_perturbation.md``
	§3.4) instead specifies a simpler, downstream-loadable variant: keep
	the existing hierarchy (same nodes, same features, same internal edges)
	and permute only the ``section → line`` leaf assignment. We follow the
	design doc because (a) it is operationally equivalent for the
	experimental claim — every line ends up in the wrong section, scene,
	act, and speaker context — and (b) it preserves x_dict compatibility
	with ``train/ablation.py`` and ``train/pretrain_structure.py`` without
	having to fabricate features for synthetic internal nodes.
	"""
	out = copy.deepcopy(data)
	g = _make_generator(seed)

	et = _find_edge_type(out, "section", "line")
	ei = out[et].edge_index
	# Permute the line column. Each section keeps the same number of children;
	# each line still appears exactly once on the line side.
	perm = torch.randperm(ei.shape[1], generator=g)
	out[et].edge_index = torch.stack([ei[0], ei[1][perm]])
	return out


PERTURBATIONS: dict[str, Callable[[HeteroData, int], HeteroData]] = {
	PerturbKind.SCENES.value: perturb_scenes,
	PerturbKind.SPEAKERS.value: perturb_speakers,
	PerturbKind.TREE.value: perturb_random_tree,
}
