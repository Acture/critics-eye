"""Tests for TEI parser fixes: cross-speech next edges and text whitespace."""

from pathlib import Path

from critics_eye.core.primary import load_primary_data

MAC_XML = Path("data/primary/shakespeare/macbeth_XML_FolgerShakespeare/Mac.xml")


def _load_macbeth():
	"""Load Macbeth graph (cached per test session via module-level would be ideal,
	but pytest fixtures handle that). For now, load fresh each time."""
	assert MAC_XML.exists(), f"Macbeth XML not found at {MAC_XML}"
	return load_primary_data(MAC_XML)


def test_line_next_edge_full_coverage():
	"""Every consecutive pair of lines should have a next edge (100% coverage)."""
	G = _load_macbeth()

	line_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "line"]
	next_line_edges = [
		(u, v)
		for u, v, d in G.edges(data=True)
		if d.get("edge_type") == "next"
		and G.nodes[u].get("node_type") == "line"
		and G.nodes[v].get("node_type") == "line"
	]

	expected = len(line_nodes) - 1
	assert (
		len(next_line_edges) == expected
	), f"Expected {expected} line→line next edges, got {len(next_line_edges)}"


def test_cross_speech_next_edge():
	"""ftln-0002 (Witch 1's last line) should connect to ftln-0003 (Witch 2's first line)."""
	G = _load_macbeth()

	next_map = {
		u: v
		for u, v, d in G.edges(data=True)
		if d.get("edge_type") == "next" and G.nodes[u].get("node_type") == "line"
	}

	assert (
		next_map.get("ftln-0002") == "ftln-0003"
	), f"Cross-speech next edge missing: ftln-0002 → {next_map.get('ftln-0002', 'MISSING')}"


def test_line_chain_walks_entire_play():
	"""Starting from ftln-0001, should be able to walk the entire play via next edges."""
	G = _load_macbeth()

	next_map = {
		u: v
		for u, v, d in G.edges(data=True)
		if d.get("edge_type") == "next"
		and G.nodes[u].get("node_type") == "line"
		and G.nodes[v].get("node_type") == "line"
	}

	num_lines = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "line")

	chain = 0
	cur = "ftln-0001"
	visited: set[str] = set()
	while cur in next_map and cur not in visited:
		visited.add(cur)
		cur = next_map[cur]
		chain += 1

	assert chain == num_lines - 1, (
		f"Chain length {chain} != expected {num_lines - 1}. " f"Chain breaks somewhere."
	)


def test_line_text_no_double_spaces():
	"""Line text should not contain double spaces (was caused by ' '.join with <c> tags)."""
	G = _load_macbeth()

	double_space_lines = []
	for n, d in G.nodes(data=True):
		if d.get("node_type") == "line":
			text = d.get("text", "")
			if "  " in text:
				double_space_lines.append((n, text[:60]))

	assert not double_space_lines, (
		f"Found {len(double_space_lines)} lines with double spaces. "
		f"First: {double_space_lines[0]}"
	)


def test_first_line_text_correct():
	"""Verify Act 1, Scene 1, Line 1 text is properly formatted."""
	G = _load_macbeth()
	text = G.nodes["ftln-0001"].get("text", "")
	assert text == "When shall we three meet again?", f"Got: {text!r}"
