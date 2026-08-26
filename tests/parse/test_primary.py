from pathlib import Path

import pytest
from networkx import MultiDiGraph

from critics_eye.core.models import KeyNodeType, StructuralEdgeType, StructuralNodeType
from critics_eye.core.primary.parsing.tei import parse


@pytest.fixture(scope="session")
def macbeth_graph(Mac_xml: Path) -> MultiDiGraph:
	return parse(Mac_xml)


def _edge_types(graph: MultiDiGraph) -> set[str]:
	return {data["edge_type"] for _, _, _, data in graph.edges(keys=True, data=True)}


def test_parse_returns_populated_graph(macbeth_graph: MultiDiGraph) -> None:
	assert isinstance(macbeth_graph, MultiDiGraph)
	assert macbeth_graph.number_of_nodes() > 0
	assert macbeth_graph.number_of_edges() > 0


def test_graph_schema(macbeth_graph: MultiDiGraph) -> None:
	node_types = [data["node_type"] for _, data in macbeth_graph.nodes(data=True)]
	assert StructuralNodeType.PLAY in node_types
	assert StructuralNodeType.ACT in node_types
	assert StructuralNodeType.SCENE in node_types
	assert StructuralNodeType.LINE in node_types
	assert KeyNodeType.CHARACTER in node_types or KeyNodeType.GROUP in node_types

	edge_types = _edge_types(macbeth_graph)
	assert StructuralEdgeType.CONTAINS in edge_types
	assert StructuralEdgeType.NEXT in edge_types
	assert StructuralEdgeType.SPOKEN_BY in edge_types


def test_macbeth_has_five_acts(macbeth_graph: MultiDiGraph) -> None:
	acts = [
		node_id
		for node_id, data in macbeth_graph.nodes(data=True)
		if data["node_type"] == StructuralNodeType.ACT
	]
	assert len(acts) == 5


def test_lines_carry_text(macbeth_graph: MultiDiGraph) -> None:
	lines = [
		data
		for _, data in macbeth_graph.nodes(data=True)
		if data["node_type"] == StructuralNodeType.LINE
	]
	assert lines
	assert any((data.get("text") or "").strip() for data in lines)


def test_speakers_resolve_to_character_nodes(macbeth_graph: MultiDiGraph) -> None:
	spoken_by = [
		(u, v)
		for u, v, d in macbeth_graph.edges(data=True)
		if d["edge_type"] == StructuralEdgeType.SPOKEN_BY
	]
	assert spoken_by
	character_ids = {
		node_id
		for node_id, data in macbeth_graph.nodes(data=True)
		if data["node_type"] in {KeyNodeType.CHARACTER, KeyNodeType.GROUP}
	}
	assert {v for _, v in spoken_by} <= character_ids


def test_all_primary_fixtures_parse(primary_xmls: list[Path]) -> None:
	assert primary_xmls
	for xml in primary_xmls:
		graph = parse(xml)
		assert graph.number_of_nodes() > 0, f"{xml.stem} produced an empty graph"
