from __future__ import annotations

import numpy as np
from networkx import MultiDiGraph

from critics_eye.core.models import KeyNodeType, StructuralNodeType, TextualNodeType
from critics_eye.utils.embedding import build_feature_matrix


class _FakeEmbeddings:
	def __init__(self, model_name: str, show_progress: bool) -> None:
		self.model_name = model_name
		self.show_progress = show_progress

	def embed_query(self, text: str) -> list[float]:
		base = float(len(text))
		return [base, base + 1.0, base + 2.0]

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		return [self.embed_query(text) for text in texts]


def test_build_feature_matrix_falls_back_for_leaf_without_text(monkeypatch) -> None:
	monkeypatch.setattr(
		"critics_eye.utils.embedding.HuggingFaceEmbeddings",
		_FakeEmbeddings,
	)

	graph = MultiDiGraph()
	graph.add_node(
		"WITCHES_Mac",
		node_id="WITCHES_Mac",
		node_reference=(None, None, None),
		node_type=KeyNodeType.GROUP,
		tag="personGrp",
		text=None,
		display_text="Three Witches",
	)
	graph.add_node(
		"line_1",
		node_id="line_1",
		node_reference=(1, 1, 1),
		node_type=TextualNodeType.LINE,
		tag="l",
		text="Fair is foul, and foul is fair.",
	)
	graph.add_node(
		"section_1",
		node_id="section_1",
		node_reference=(1, 1, None),
		node_type=StructuralNodeType.SECTION,
		tag="speech",
		text=None,
	)
	graph.add_edge("section_1", "line_1", edge_type="contains")

	node_order, feature_matrix = build_feature_matrix(graph)

	assert node_order == ["WITCHES_Mac", "line_1", "section_1"]
	assert feature_matrix.shape == (3, 3)
	assert np.isfinite(feature_matrix).all()
	assert feature_matrix[0, 0] == float(len("Three Witches"))
