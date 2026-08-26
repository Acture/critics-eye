from pathlib import Path

from networkx import MultiDiGraph

from critics_eye.core.models import KeyNodeType, StructuralEdgeType, StructuralNodeType
from critics_eye.core.primary.parsing.tei_dracor import parse

DRACOR_DIR = Path("tests/fixture/dracor")
FIXTURES = [
	"a-the-valiant-welshman.xml",
	"a-midsummer-nights-dream.xml",
	"achat-ein-april-scherz.xml",
	"z-verse-and-prose-containers.xml",
]


def _edge_types(graph: MultiDiGraph) -> set[str]:
	return {data["edge_type"] for _, _, _, data in graph.edges(keys=True, data=True)}


def test_dracor_fixtures_parse() -> None:
	for fixture in FIXTURES:
		graph = parse(DRACOR_DIR / fixture)
		assert isinstance(graph, MultiDiGraph)
		assert graph.number_of_nodes() > 0
		assert graph.number_of_edges() > 0


def test_dracor_graph_schema_and_speaker_resolution() -> None:
	for fixture in FIXTURES:
		graph = parse(DRACOR_DIR / fixture)
		node_types = [data["node_type"] for _, data in graph.nodes(data=True)]
		assert StructuralNodeType.PLAY in node_types
		assert StructuralNodeType.SCENE in node_types
		assert StructuralNodeType.SECTION in node_types
		assert StructuralNodeType.LINE in node_types
		assert KeyNodeType.CHARACTER in node_types or KeyNodeType.GROUP in node_types
		assert any((data.get("text") or "").strip() for _, data in graph.nodes(data=True) if data["node_type"] == StructuralNodeType.LINE)

		edge_types = _edge_types(graph)
		assert StructuralEdgeType.CONTAINS in edge_types
		assert StructuralEdgeType.NEXT in edge_types
		assert StructuralEdgeType.SPOKEN_BY in edge_types
		assert StructuralEdgeType.APPEARS_IN in edge_types


def test_shakedracor_has_act_nodes_and_spoken_by_edges() -> None:
	graph = parse(DRACOR_DIR / "a-midsummer-nights-dream.xml")
	act_nodes = [node_id for node_id, data in graph.nodes(data=True) if data["node_type"] == StructuralNodeType.ACT]
	assert len(act_nodes) == 5
	spoken_by_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d["edge_type"] == StructuralEdgeType.SPOKEN_BY]
	assert any(v == "Theseus_MND" for _, v, _ in spoken_by_edges)


def test_engdracor_line_ids_are_synthesized_when_missing() -> None:
	graph = parse(DRACOR_DIR / "a-the-valiant-welshman.xml")
	synth_line_ids = [node_id for node_id, data in graph.nodes(data=True) if data["node_type"] == StructuralNodeType.LINE and node_id.startswith("l_")]
	assert synth_line_ids
	assert graph.nodes[synth_line_ids[0]]["text"]


def test_gerdracor_prose_paragraphs_become_line_nodes() -> None:
	graph = parse(DRACOR_DIR / "achat-ein-april-scherz.xml")
	line_nodes = [(node_id, data) for node_id, data in graph.nodes(data=True) if data["node_type"] == StructuralNodeType.LINE]
	assert len(line_nodes) >= 100
	assert any(data["text"] == "Betty!" for _, data in line_nodes)
	spoken_by_targets = [v for u, v, d in graph.edges(data=True) if d["edge_type"] == StructuralEdgeType.SPOKEN_BY and u.startswith("sp_")]
	assert "fraeulein" in spoken_by_targets


def test_bare_verse_and_prose_under_a_scene_are_parsed() -> None:
	"""Verse drama hangs choruses as bare <lg> under a scene and prose drama does
	the same with <p>; neither is wrapped in <sp>. Rejecting them made Faust,
	Nathan der Weise and Kabale und Liebe unparseable."""
	graph = parse(DRACOR_DIR / "z-verse-and-prose-containers.xml")
	texts = {(data.get("text") or "").strip() for _, data in graph.nodes(data=True) if data["node_type"] == StructuralNodeType.LINE}
	assert "Ihr naht euch wieder, schwankende Gestalten," in texts
	assert "Ein Chor ohne Sprecher, direkt unter der Szene." in texts
	assert "Ein zweiter unnumerierter Auftritt: ohne n, ohne xml:id." in texts


def test_scene_ids_stay_distinct_when_numbers_repeat_across_acts() -> None:
	"""Scene numbers restart every act, and unnumbered scenes carry neither `n`
	nor `xml:id`, so naive ids collided and silently merged separate scenes."""
	graph = parse(DRACOR_DIR / "z-verse-and-prose-containers.xml")
	scenes = [node_id for node_id, data in graph.nodes(data=True) if data["node_type"] == StructuralNodeType.SCENE]
	assert len(scenes) == 5
	assert len(set(scenes)) == 5


def test_label_captions_are_skipped_not_rejected() -> None:
	"""<label> titles an inset song the way <head> titles a scene; it made The
	Tempest unparseable."""
	graph = parse(DRACOR_DIR / "z-verse-and-prose-containers.xml")
	texts = {(data.get("text") or "").strip() for _, data in graph.nodes(data=True)}
	assert "Gesang." not in texts


def test_stage_directions_use_the_same_node_type_as_the_folger_parser() -> None:
	"""The adaptor exists to emit graphs the rest of the pipeline can read, so a
	stage direction must not be a SECTION here and a STAGE_DIRECTION there."""
	graph = parse(DRACOR_DIR / "a-midsummer-nights-dream.xml")
	node_types = [data["node_type"] for _, data in graph.nodes(data=True)]
	assert StructuralNodeType.STAGE_DIRECTION in node_types


def test_german_ordinal_headings_yield_act_and_scene_numbers() -> None:
	"""German editions come from TextGrid reading texts and carry no `@n`; the
	number exists only as a word in the <head>. Without reading it, every German
	act and scene parses with an empty reference and cannot be aligned."""
	graph = parse(DRACOR_DIR / "z-verse-and-prose-containers.xml")
	acts = {data["node_reference"][0] for _, data in graph.nodes(data=True) if data["node_type"] == StructuralNodeType.ACT}
	assert acts == {1, 2}
	scenes = {data["node_reference"][:2] for _, data in graph.nodes(data=True) if data["node_type"] == StructuralNodeType.SCENE}
	assert (1, 1) in scenes
	assert (2, 1) in scenes
