"""Tests for the visitor-pattern TEI parser."""
from pathlib import Path

import pytest
from networkx import MultiDiGraph

from critics_eye.core.primary.parsing.tei import parse

TEI_BASE = Path("data/primary/shakespeare")

# (stem, min_nodes, min_edges) — conservative lower bounds
PLAYS = [
	("macbeth_XML_FolgerShakespeare/Mac", 3000, 6000),
	("hamlet_XML_FolgerShakespeare/Ham", 5000, 10000),
	("othello_XML_FolgerShakespeare/Oth", 5000, 10000),
	("a-midsummer-nights-dream_XML_FolgerShakespeare/MND", 3000, 6000),
]

ALL_DRAMAS = [
	d / xmls[0].name
	for d in sorted(TEI_BASE.iterdir())
	if (xmls := list(d.glob("*.xml")))
	and xmls[0].stem not in {"Son", "Luc", "PhT", "Ven"}
]


@pytest.fixture
def xml_path(request: pytest.FixtureRequest) -> Path:
	rel = request.param
	p = TEI_BASE / (rel + ".xml")
	if not p.exists():
		pytest.skip(f"TEI file not found: {p}")
	return p


@pytest.mark.parametrize("xml_path", [p[0] for p in PLAYS], indirect=True)
def test_target_plays_parse(xml_path: Path) -> None:
	"""Target plays (Mac, Ham, Oth, MND) parse without error."""
	graph = parse(xml_path)
	assert isinstance(graph, MultiDiGraph)
	assert graph.number_of_nodes() > 0
	assert graph.number_of_edges() > 0


@pytest.mark.parametrize("xml_path", [p[0] for p in PLAYS], indirect=True)
def test_target_plays_node_counts(xml_path: Path) -> None:
	"""Target plays produce graphs with reasonable node/edge counts."""
	graph = parse(xml_path)
	expected = {p[0].split("/")[-1]: (p[1], p[2]) for p in PLAYS}
	stem = xml_path.stem
	min_nodes, min_edges = expected[stem]
	assert graph.number_of_nodes() >= min_nodes, f"{stem}: expected >= {min_nodes} nodes, got {graph.number_of_nodes()}"
	assert graph.number_of_edges() >= min_edges, f"{stem}: expected >= {min_edges} edges, got {graph.number_of_edges()}"


@pytest.mark.parametrize("xml_path", ALL_DRAMAS, ids=[p.stem for p in ALL_DRAMAS])
def test_all_dramas_parse(xml_path: Path) -> None:
	"""All 38 Shakespeare dramas parse without crashing."""
	if not xml_path.exists():
		pytest.skip(f"TEI file not found: {xml_path}")
	graph = parse(xml_path)
	assert isinstance(graph, MultiDiGraph)
	assert graph.number_of_nodes() > 100
	assert graph.number_of_edges() > 200


def test_macbeth_has_expected_node_types() -> None:
	"""Macbeth graph contains expected node types."""
	mac = TEI_BASE / "macbeth_XML_FolgerShakespeare" / "Mac.xml"
	if not mac.exists():
		pytest.skip("Mac.xml not found")
	graph = parse(mac)

	node_types = {data.get("node_type") for _, data in graph.nodes(data=True)}
	assert "play" in node_types
	assert "act" in node_types
	assert "scene" in node_types
	assert "section" in node_types
	assert "line" in node_types
	assert "character" in node_types


def test_hamlet_edge_case_no_xmlid() -> None:
	"""Hamlet has one <sp> without xml:id — parser should synthesize an ID."""
	ham = TEI_BASE / "hamlet_XML_FolgerShakespeare" / "Ham.xml"
	if not ham.exists():
		pytest.skip("Ham.xml not found")
	graph = parse(ham)
	# The synthesized ID should start with "sp_"
	synth_ids = [n for n in graph.nodes if n.startswith("sp_")]
	assert len(synth_ids) >= 1, "Expected at least one synthesized sp_ node ID for Hamlet"


def test_macbeth_multi_ref_header_does_not_create_empty_placeholder_nodes() -> None:
	"""Multi-target `ref` attributes should split into separate edges, not empty nodes."""
	mac = TEI_BASE / "macbeth_XML_FolgerShakespeare" / "Mac.xml"
	if not mac.exists():
		pytest.skip("Mac.xml not found")
	graph = parse(mac)

	assert "Macduff_Mac #LadyMacduff_Mac" not in graph.nodes
	empty_nodes = [node_id for node_id, data in graph.nodes(data=True) if not data]
	assert not empty_nodes, f"Unexpected placeholder nodes: {empty_nodes[:5]}"


def test_macbeth_display_text_prefers_source_text_and_falls_back_conservatively() -> None:
	"""Display text should come from TEI text when available, else fall back to raw IDs."""
	mac = TEI_BASE / "macbeth_XML_FolgerShakespeare" / "Mac.xml"
	if not mac.exists():
		pytest.skip("Mac.xml not found")
	graph = parse(mac)

	assert graph.nodes["LadyMacbeth_Mac"]["display_text"] == "Lady Macbeth"
	assert graph.nodes["WITCHES_Mac"]["display_text"] == "Three Witches, the Weïrd Sisters"
	assert graph.nodes["MESSENGERS_Mac"]["display_text"] == "MESSENGERS_Mac"
