"""Unit tests for ``critics_eye.utils.perturb``."""

from __future__ import annotations

import copy

import pytest
import torch
from torch_geometric.data import HeteroData

from critics_eye.utils.perturb import (
	PERTURBATIONS,
	PerturbKind,
	perturb_random_tree,
	perturb_scenes,
	perturb_speakers,
)


# ---------------------------------------------------------------------------
# Fixture: a small synthetic Macbeth-shaped HeteroData
# ---------------------------------------------------------------------------
#
# Hierarchy:
#   1 play
#   2 acts (a0, a1)
#   4 scenes (s0,s1 in a0; s2,s3 in a1)
#   8 sections (sec0..sec3 under s0..s3 — 2 sections per scene; one section
#               per scene is a parent of one extra child section, giving
#               4 + 4 = 8 leaves attached to 16 lines)
#   16 lines (2 per leaf section)
#   3 characters, 2 groups
# ---------------------------------------------------------------------------


def _make_fixture() -> HeteroData:
	d = HeteroData()
	# Node features (just unique identifiers per node, dim=4).
	d["play"].x = torch.randn(1, 4)
	d["act"].x = torch.randn(2, 4)
	d["scene"].x = torch.randn(4, 4)
	d["section"].x = torch.randn(8, 4)
	d["line"].x = torch.randn(16, 4)
	d["character"].x = torch.randn(3, 4)
	d["group"].x = torch.randn(2, 4)

	# play → act
	d["play", "contains", "act"].edge_index = torch.tensor(
		[[0, 0], [0, 1]], dtype=torch.long
	)
	# act → scene
	d["act", "contains", "scene"].edge_index = torch.tensor(
		[[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long
	)
	# scene → section (each scene owns 2 sections)
	d["scene", "contains", "section"].edge_index = torch.tensor(
		[[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long
	)
	# section → line (each section owns 2 lines)
	src = torch.arange(8).repeat_interleave(2)
	dst = torch.arange(16)
	d["section", "contains", "line"].edge_index = torch.stack([src, dst])
	# line → line: complete chain across all 16 lines
	d["line", "next", "line"].edge_index = torch.stack(
		[torch.arange(15), torch.arange(1, 16)]
	)
	# section → character (every section spoken by some character)
	d["section", "spoken_by", "character"].edge_index = torch.tensor(
		[[0, 1, 2, 3, 4, 5], [0, 1, 0, 2, 1, 2]], dtype=torch.long
	)
	# section → group (a couple of sections spoken by groups)
	d["section", "spoken_by", "group"].edge_index = torch.tensor(
		[[6, 7], [0, 1]], dtype=torch.long
	)
	return d


def _node_counts(data: HeteroData) -> dict[str, int]:
	return {nt: data[nt].num_nodes for nt in data.node_types}


def _edge_counts(data: HeteroData) -> dict[tuple, int]:
	return {et: data[et].edge_index.shape[1] for et in data.edge_types}


# ---------------------------------------------------------------------------
# Generic invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(PerturbKind))
def test_node_count_invariance(kind: PerturbKind) -> None:
	d = _make_fixture()
	before = _node_counts(d)
	out = PERTURBATIONS[kind.value](d, seed=42)
	assert _node_counts(out) == before


@pytest.mark.parametrize("kind", list(PerturbKind))
def test_edge_count_invariance(kind: PerturbKind) -> None:
	d = _make_fixture()
	before = _edge_counts(d)
	out = PERTURBATIONS[kind.value](d, seed=42)
	assert _edge_counts(out) == before


@pytest.mark.parametrize("kind", list(PerturbKind))
def test_does_not_mutate_input(kind: PerturbKind) -> None:
	d = _make_fixture()
	snapshot_edges = {et: d[et].edge_index.clone() for et in d.edge_types}
	_ = PERTURBATIONS[kind.value](d, seed=7)
	for et, ei in snapshot_edges.items():
		assert torch.equal(d[et].edge_index, ei), f"input mutated for {et}"


@pytest.mark.parametrize("kind", list(PerturbKind))
def test_reproducibility(kind: PerturbKind) -> None:
	d = _make_fixture()
	a = PERTURBATIONS[kind.value](d, seed=42)
	b = PERTURBATIONS[kind.value](copy.deepcopy(d), seed=42)
	for et in d.edge_types:
		assert torch.equal(
			a[et].edge_index, b[et].edge_index
		), f"non-deterministic edges for {et}"


@pytest.mark.parametrize("kind", list(PerturbKind))
def test_different_seeds_differ(kind: PerturbKind) -> None:
	d = _make_fixture()
	a = PERTURBATIONS[kind.value](d, seed=0)
	b = PERTURBATIONS[kind.value](d, seed=1)
	# At least one edge type must differ across distinct seeds.
	assert any(
		not torch.equal(a[et].edge_index, b[et].edge_index) for et in d.edge_types
	)


# ---------------------------------------------------------------------------
# P1 — scenes
# ---------------------------------------------------------------------------


def test_p1_preserves_line_next_chain() -> None:
	d = _make_fixture()
	out = perturb_scenes(d, seed=3)
	assert torch.equal(
		out["line", "next", "line"].edge_index,
		d["line", "next", "line"].edge_index,
	)


def test_p1_preserves_act_to_scene() -> None:
	d = _make_fixture()
	out = perturb_scenes(d, seed=3)
	assert torch.equal(
		out["act", "contains", "scene"].edge_index,
		d["act", "contains", "scene"].edge_index,
	)


def test_p1_keeps_within_act() -> None:
	"""Each section must still end up under a scene that belongs to the same act."""
	d = _make_fixture()
	out = perturb_scenes(d, seed=11)

	# scene → act lookup (first act per scene; in our fixture each scene has
	# exactly one act anyway).
	a2s = d["act", "contains", "scene"].edge_index
	act_of_scene = torch.full((d["scene"].num_nodes,), -1, dtype=torch.long)
	for a, s in zip(a2s[0].tolist(), a2s[1].tolist()):
		if act_of_scene[s] == -1:
			act_of_scene[s] = a

	orig = d["scene", "contains", "section"].edge_index
	new = out["scene", "contains", "section"].edge_index

	# Map section → original act, section → new act, must match per section.
	orig_act_of_sec: dict[int, int] = {}
	for s, sec in zip(orig[0].tolist(), orig[1].tolist()):
		orig_act_of_sec[sec] = int(act_of_scene[s].item())

	for s, sec in zip(new[0].tolist(), new[1].tolist()):
		assert (
			int(act_of_scene[s].item()) == orig_act_of_sec[sec]
		), f"section {sec} moved across acts"


def test_p1_actually_shuffles() -> None:
	"""With our fixture (2 sections per scene, 2 scenes per act), seed=0 should
	produce a non-identity permutation in at least one act."""
	d = _make_fixture()
	out = perturb_scenes(d, seed=0)
	assert not torch.equal(
		out["scene", "contains", "section"].edge_index,
		d["scene", "contains", "section"].edge_index,
	)


# ---------------------------------------------------------------------------
# P2 — speakers
# ---------------------------------------------------------------------------


def test_p2_section_count_unchanged() -> None:
	d = _make_fixture()
	out = perturb_speakers(d, seed=5)
	assert out["section"].num_nodes == d["section"].num_nodes


def test_p2_character_count_unchanged() -> None:
	d = _make_fixture()
	out = perturb_speakers(d, seed=5)
	assert out["character"].num_nodes == d["character"].num_nodes
	assert out["group"].num_nodes == d["group"].num_nodes


def test_p2_spoken_by_count_unchanged() -> None:
	d = _make_fixture()
	out = perturb_speakers(d, seed=5)
	assert (
		out["section", "spoken_by", "character"].edge_index.shape[1]
		== d["section", "spoken_by", "character"].edge_index.shape[1]
	)
	assert (
		out["section", "spoken_by", "group"].edge_index.shape[1]
		== d["section", "spoken_by", "group"].edge_index.shape[1]
	)


def test_p2_section_ids_preserved() -> None:
	"""Source column (section ids) must be byte-identical; only target permuted."""
	d = _make_fixture()
	out = perturb_speakers(d, seed=5)
	for et in (
		("section", "spoken_by", "character"),
		("section", "spoken_by", "group"),
	):
		assert torch.equal(out[et].edge_index[0], d[et].edge_index[0])


def test_p2_speaker_multiset_preserved() -> None:
	"""Per-character speech count is preserved (we only permute targets)."""
	d = _make_fixture()
	out = perturb_speakers(d, seed=5)
	for et in (
		("section", "spoken_by", "character"),
		("section", "spoken_by", "group"),
	):
		orig = torch.bincount(d[et].edge_index[1])
		new = torch.bincount(out[et].edge_index[1])
		assert torch.equal(orig, new)


# ---------------------------------------------------------------------------
# P4 — random tree leaves
# ---------------------------------------------------------------------------


def test_p4_section_to_line_count_unchanged() -> None:
	d = _make_fixture()
	out = perturb_random_tree(d, seed=9)
	assert (
		out["section", "contains", "line"].edge_index.shape[1]
		== d["section", "contains", "line"].edge_index.shape[1]
	)


def test_p4_hierarchy_depth_preserved() -> None:
	"""The internal hierarchy (play→act→scene→section) is untouched."""
	d = _make_fixture()
	out = perturb_random_tree(d, seed=9)
	for et in (
		("play", "contains", "act"),
		("act", "contains", "scene"),
		("scene", "contains", "section"),
	):
		assert torch.equal(out[et].edge_index, d[et].edge_index)


def test_p4_each_line_attached_once() -> None:
	d = _make_fixture()
	out = perturb_random_tree(d, seed=9)
	lines = out["section", "contains", "line"].edge_index[1]
	assert lines.unique().numel() == d["line"].num_nodes


def test_p4_per_section_lines_count_preserved() -> None:
	d = _make_fixture()
	out = perturb_random_tree(d, seed=9)
	orig = torch.bincount(
		d["section", "contains", "line"].edge_index[0],
		minlength=d["section"].num_nodes,
	)
	new = torch.bincount(
		out["section", "contains", "line"].edge_index[0],
		minlength=d["section"].num_nodes,
	)
	assert torch.equal(orig, new)
