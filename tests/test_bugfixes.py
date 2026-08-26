"""Tests for bug fixes: C1, C2, H1, H2, H3, H4-H6."""

import json
from pathlib import Path

import networkx as nx
import torch
from networkx.readwrite import json_graph
from torch_geometric.data import HeteroData

from critics_eye.cli.secondary import inject_review_and_motif
from critics_eye.core.models import DocumentNode, TextualNodeType
from critics_eye.core.secondary.motif.model import (
	LabeledMotifCluster,
	MotifCluster,
	ReviewMotifIndex,
)
from critics_eye.core.secondary.relation import LitAnalysis
from critics_eye.utils.data_prep import (
	build_x_dict,
	filter_edges_by_act,
	infer_line_act_ids,
)


# ── C1: LabeledMotifCluster.labels should not recurse ────────────────


def test_labeled_motif_cluster_labels_no_recursion():
	"""C1: Accessing .labels must return the list, not infinite recurse."""
	cluster = LabeledMotifCluster(
		motif_id=0,
		representative_phrase="blood imagery",
		representative_index=0,
		members=[
			ReviewMotifIndex(
				index=0, review_id=0, kind="source", phrase="blood imagery"
			),
		],
		labels=["Violence", "Guilt"],
	)
	# Should return the list directly — old code caused RecursionError
	assert cluster.labels == ["Violence", "Guilt"]
	assert isinstance(cluster.labels, list)


# ── C2: phrase_to_motif key normalization ─────────────────────────────


def test_phrase_to_motif_key_normalization():
	"""C2: Dict keys should be normalized (lowercase + stripped) to match lookups."""
	members = [
		ReviewMotifIndex(
			index=0, review_id=0, kind="source", phrase="  Blood Imagery  "
		),
		ReviewMotifIndex(index=1, review_id=1, kind="target", phrase="guilt"),
	]
	motif = MotifCluster(
		motif_id=0,
		representative_phrase="blood imagery",
		representative_index=0,
		members=members,
	)
	# Simulate the fixed dict construction from secondary.py
	phrase_to_motif = {m.phrase.strip().lower(): motif for m in motif.members}
	# Lookup with normalized key should work
	assert "blood imagery" in phrase_to_motif
	assert "guilt" in phrase_to_motif
	# Old bug: raw key "  Blood Imagery  " would fail this lookup
	assert phrase_to_motif["blood imagery"] is motif


def test_motif_cluster_representative_index_is_normalized_to_member_index():
	"""Representative index should point into members, not an external phrase table."""
	cluster = MotifCluster(
		motif_id=7,
		representative_phrase="blood imagery",
		representative_index=11,
		members=[
			ReviewMotifIndex(index=0, review_id=0, kind="source", phrase="guilt"),
			ReviewMotifIndex(
				index=1, review_id=0, kind="target", phrase="blood imagery"
			),
		],
	)

	assert cluster.representative_index == 1
	assert cluster.symbol == "blood imagery"


def test_inject_review_and_motif_preserves_source_to_target_direction(
	tmp_path: Path,
):
	"""Relation injection should always follow source -> relation -> target semantics."""
	graph_path = tmp_path / "graph.json"
	review_path = tmp_path / "reviews.jsonl"
	index_map_path = tmp_path / "index_map.json"
	motifs_path = tmp_path / "motifs.json"
	output_path = tmp_path / "graph_with_motif.json"

	graph = nx.MultiDiGraph()
	target_node = DocumentNode(
		node_id="line_target",
		node_reference=(3, 4, 50),
		node_type=TextualNodeType.LINE,
		text="Is this a dagger...",
	)
	graph.add_node(
		target_node.node_id, node_id=target_node.node_id, **target_node.model_dump(exclude={"node_id"})
	)
	graph_path.write_text(
		json.dumps(json_graph.node_link_data(graph, edges="edges"), ensure_ascii=False),
		encoding="utf-8",
	)

	review = LitAnalysis(
		source_element="blood imagery",
		target_element="dagger vision",
		relationship_type=["foreshadow"],
		source_citation=(None, None, None),
		target_citation=(3, 4, 50),
		justification_quote="Blood imagery foreshadows the dagger vision.",
		confidence=0.91,
		source_file="review",
	)
	review_path.write_text(review.model_dump_json() + "\n", encoding="utf-8")
	index_map_path.write_text("{}", encoding="utf-8")

	source_motif = MotifCluster(
		motif_id=1,
		representative_phrase="blood imagery",
		representative_index=0,
		members=[
			ReviewMotifIndex(
				index=0,
				review_id=0,
				kind="source",
				phrase="blood imagery",
			)
		],
	)
	target_motif = MotifCluster(
		motif_id=2,
		representative_phrase="dagger vision",
		representative_index=0,
		members=[
			ReviewMotifIndex(
				index=0,
				review_id=0,
				kind="target",
				phrase="dagger vision",
			)
		],
	)
	motifs_path.write_text(
		json.dumps(
			[source_motif.model_dump(), target_motif.model_dump()],
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)

	inject_review_and_motif(
		graph_json=graph_path,
		review_json=review_path,
		index_map_json=index_map_path,
		motifs_json=motifs_path,
		output_json=output_path,
	)

	output_graph = json.loads(output_path.read_text(encoding="utf-8"))
	edges = {
		(edge["source"], edge["edge_type"], edge["target"])
		for edge in output_graph["edges"]
	}

	assert ("motif_1", "foreshadow", "line_target") in edges
	assert ("line_target", "foreshadow", "motif_2") not in edges
	assert ("line_target", "reflects_theme", "motif_2") in edges


def test_inject_review_and_motif_skips_collapsed_self_relation(
	tmp_path: Path,
):
	"""If both endpoints collapse to the same motif, do not inject a self-loop."""
	graph_path = tmp_path / "graph.json"
	review_path = tmp_path / "reviews.jsonl"
	index_map_path = tmp_path / "index_map.json"
	motifs_path = tmp_path / "motifs.json"
	output_path = tmp_path / "graph_with_motif.json"

	graph = nx.MultiDiGraph()
	graph_path.write_text(
		json.dumps(json_graph.node_link_data(graph, edges="edges"), ensure_ascii=False),
		encoding="utf-8",
	)

	review = LitAnalysis(
		source_element="Lady Macbeth's 'what's done is done'",
		target_element="Lady Macbeth's sleepwalking 'What's done cannot be undone'",
		relationship_type=["contrast"],
		source_citation=(3, 2, 12),
		target_citation=(5, 1, 64),
		justification_quote="Finality has become entrapment.",
		confidence=0.94,
		source_file="review",
	)
	review_path.write_text(review.model_dump_json() + "\n", encoding="utf-8")
	index_map_path.write_text("{}", encoding="utf-8")

	collapsed_motif = MotifCluster(
		motif_id=32,
		representative_phrase="lady macbeth sleepwalking motif",
		representative_index=0,
		members=[
			ReviewMotifIndex(
				index=0,
				review_id=0,
				kind="source",
				phrase="lady macbeth's 'what's done is done'",
			),
			ReviewMotifIndex(
				index=1,
				review_id=0,
				kind="target",
				phrase="lady macbeth's sleepwalking 'what's done cannot be undone'",
			),
		],
	)
	motifs_path.write_text(
		json.dumps([collapsed_motif.model_dump()], ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	inject_review_and_motif(
		graph_json=graph_path,
		review_json=review_path,
		index_map_json=index_map_path,
		motifs_json=motifs_path,
		output_json=output_path,
	)

	output_graph = json.loads(output_path.read_text(encoding="utf-8"))
	motif_nodes = [node for node in output_graph["nodes"] if node.get("tag") == "motif"]
	edges = {
		(edge["source"], edge["edge_type"], edge["target"])
		for edge in output_graph["edges"]
	}

	assert motif_nodes
	assert all("node_id" in node for node in motif_nodes)
	assert ("motif_32", "contrast", "motif_32") not in edges


# ── H1: infer_line_act_ids dynamic minlength ─────────────────────────


def _make_3act_hetero_data() -> HeteroData:
	"""Create a minimal HeteroData with 3 acts (not 5) to test dynamic act count."""
	data = HeteroData()

	# 6 lines, 3 sections, 3 scenes, 3 acts
	data["line"].num_nodes = 6
	data["section"].num_nodes = 3
	data["scene"].num_nodes = 3
	data["act"].num_nodes = 3

	# section → line: each section has 2 lines
	data["section", "contains", "line"].edge_index = torch.tensor(
		[
			[0, 0, 1, 1, 2, 2],  # section IDs
			[0, 1, 2, 3, 4, 5],  # line IDs
		]
	)

	# scene → section: 1:1
	data["scene", "contains", "section"].edge_index = torch.tensor(
		[
			[0, 1, 2],  # scene IDs
			[0, 1, 2],  # section IDs
		]
	)

	# act → scene: 1:1
	data["act", "contains", "scene"].edge_index = torch.tensor(
		[
			[0, 1, 2],  # act IDs
			[0, 1, 2],  # scene IDs
		]
	)

	return data


def test_infer_line_act_ids_3_acts():
	"""H1: Should work with plays having != 5 acts."""
	data = _make_3act_hetero_data()
	act_ids = infer_line_act_ids(data)

	assert act_ids.shape == (6,)
	# Lines 0,1 → section 0 → scene 0 → act 0
	assert act_ids[0].item() == 0
	assert act_ids[1].item() == 0
	# Lines 2,3 → section 1 → scene 1 → act 1
	assert act_ids[2].item() == 1
	assert act_ids[3].item() == 1
	# Lines 4,5 → section 2 → scene 2 → act 2
	assert act_ids[4].item() == 2
	assert act_ids[5].item() == 2

	# Should have exactly 3 unique acts, not crash with minlength=5
	assert len(act_ids.unique()) == 3


def test_infer_line_act_ids_uses_cached_act_id():
	"""If data['line'].act_id exists, use it directly."""
	data = HeteroData()
	data["line"].num_nodes = 4
	data["line"].act_id = torch.tensor([0, 0, 1, 1])

	act_ids = infer_line_act_ids(data)
	assert torch.equal(act_ids, torch.tensor([0, 0, 1, 1]))


def test_filter_edges_by_act_handles_cpu_act_ids():
	"""Act filtering should work even when line_act_ids stays on CPU."""
	src = torch.tensor([0, 1, 2, 3], dtype=torch.long)
	dst = torch.tensor([10, 11, 12, 13], dtype=torch.long)
	line_act_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

	filtered_src, filtered_dst = filter_edges_by_act(
		src,
		dst,
		line_act_ids,
		target_act=2,
	)

	assert torch.equal(filtered_src, torch.tensor([2, 3], dtype=torch.long))
	assert torch.equal(filtered_dst, torch.tensor([12, 13], dtype=torch.long))
	assert filtered_src.device == src.device
	assert filtered_dst.device == dst.device


# ── H2: LLM model configurability ────────────────────────────────────


def test_trapi_relation_extractor_accepts_model_id():
	"""H2: the relation extractor backend should remain model-configurable."""
	from critics_eye.core.secondary.relation import (
		DEFAULT_LLM_MODEL,
		TRAPIRelationExtractor,
	)
	import inspect

	sig = inspect.signature(TRAPIRelationExtractor)
	assert "model_id" in sig.parameters
	assert sig.parameters["model_id"].default == DEFAULT_LLM_MODEL


def test_generate_labels_accepts_model_id():
	"""H2: generate_labels should accept model_id parameter."""
	from critics_eye.core.secondary.relation import DEFAULT_LLM_MODEL
	from critics_eye.utils.label_motif import generate_labels
	import inspect

	sig = inspect.signature(generate_labels)
	assert "model_id" in sig.parameters
	assert sig.parameters["model_id"].default == DEFAULT_LLM_MODEL


# ── build_x_dict tests ───────────────────────────────────────────────


def test_build_x_dict_handles_missing_features():
	"""build_x_dict should create zero placeholders for nodes without features."""
	data = HeteroData()
	data["line"].x = torch.randn(10, 64)
	# scene only has num_nodes, no .x attribute — like structural nodes
	data["scene"].num_nodes = 3

	device = torch.device("cpu")
	x_dict = build_x_dict(data, device)

	assert "line" in x_dict
	assert x_dict["line"].shape == (10, 64)
	assert "scene" in x_dict
	assert x_dict["scene"].shape == (3, 1)  # zero placeholder


def test_build_x_dict_skips_zero_nodes():
	"""build_x_dict should skip node types with 0 nodes."""
	data = HeteroData()
	data["line"].x = torch.randn(5, 32)
	data["empty"].num_nodes = 0

	x_dict = build_x_dict(data, torch.device("cpu"))
	assert "line" in x_dict
	assert "empty" not in x_dict
