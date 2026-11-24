"""Export cleaned MotifBench Macbeth dataset, full play TXT/PDF, and audit report for Harbor."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = (
	ROOT
	/ "research_store/data/processed/shakespeare/macbeth/published/Mac_with_motif.json"
)
DEFAULT_OUTPUT = ROOT / "research_store/output/motifbench_harbor/macbeth"
HF_REVISION = "fa060c5ad43ea3fdccb7cb6ad02a7bc85fa20c70"
HF_MACBETH_URL = (
	f"https://huggingface.co/datasets/Acture/MotifBench/resolve/{HF_REVISION}"
	"/macbeth/macbeth.jsonl"
)

ROMAN_ACT = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}

SPEAKER_OVERRIDES: dict[str, str] = {
	"ATTENDANTS.0_Mac": "ATTENDANT",
	"ATTENDANTS_Mac": "ATTENDANTS",
	"Doctor1_Mac": "ENGLISH DOCTOR",
	"Doctor2_Mac": "SCOTTISH DOCTOR",
	"LadyMacbeth_Mac": "LADY MACBETH",
	"LadyMacduff_Mac": "LADY MACDUFF",
	"MacduffsSon_Mac": "MACDUFF'S SON",
	"MESSENGERS.1_Mac": "MESSENGER 1",
	"MESSENGERS.2_Mac": "MESSENGER 2",
	"MESSENGERS.3_Mac": "MESSENGER 3",
	"MURDERERS.0.1_Mac": "MURDERER",
	"MURDERERS.1_Mac": "FIRST MURDERER",
	"MURDERERS.2_Mac": "SECOND MURDERER",
	"MURDERERS.3_Mac": "THIRD MURDERER",
	"MURDERERS_Mac": "MURDERERS",
	"OldMan_Mac": "OLD MAN",
	"SERVANTS.X.1_Mac": "SERVANT 1",
	"SERVANTS.X.2_Mac": "SERVANT 2",
	"SERVANTS.X.3_Mac": "SERVANT 3",
	"SERVANTS.X.4_Mac": "SERVANT 4",
	"SERVANTS_Mac": "SERVANT",
	"SOLDIERS.Captain_Mac": "CAPTAIN",
	"SOLDIERS.MACBETH_Mac": "MACBETH'S SOLDIERS",
	"SOLDIERS.MALCOLM.0.1_Mac": "MALCOLM'S SOLDIER",
	"SOLDIERS.MALCOLM_Mac": "MALCOLM'S SOLDIERS",
	"SPIRITS.1_Mac": "FIRST APPARITION",
	"SPIRITS.2_Mac": "SECOND APPARITION",
	"SPIRITS.3_Mac": "THIRD APPARITION",
	"SPIRITS.KINGS_Mac": "SHOW OF KINGS",
	"WITCHES.1_Mac": "FIRST WITCH",
	"WITCHES.2_Mac": "SECOND WITCH",
	"WITCHES.3_Mac": "THIRD WITCH",
	"YoungSiward_Mac": "YOUNG SIWARD",
}

CITE_RE = re.compile(
	r"\b\d+\.\d+(?:\.\d+)?\b"
	r"|\b(?:act|scene|sc\.|line|lines|ll?\.)\s*[\divx]+\b"
	r"|\bftln-\d+\b",
	re.IGNORECASE,
)

POINTER_RE = re.compile(
	r"^(?:the\s+)?(?:line|lines|passage|speech|soliloquy|pronoun|word|phrase|quote|utterance|opening|close|closing)\b"
	r"|\b(?:line|lines|passage|speech|soliloquy)\s+[\"\'\u201c\u2018]"
	r"|[\"\'\u201d\u2019]\s+(?:line|lines|speech|soliloquy|passage)\b"
	r"|\b(?:line|lines|passage|speech|soliloquy)$"
	r"|\blines\s+that\s+literally\b",
	re.IGNORECASE,
)

PURE_QUOTE_RE = re.compile(
	r"^[\"\'\u201c\u2018].*[\"\'\u201d\u2019]\s*(?:line|lines|speech|passage|soliloquy|reference|advice)?$",
	re.IGNORECASE,
)

STOPWORDS: frozenset[str] = frozenset(
	{
		"the",
		"and",
		"that",
		"this",
		"with",
		"from",
		"have",
		"will",
		"what",
		"when",
		"your",
		"their",
		"there",
		"into",
		"been",
		"were",
		"they",
		"them",
		"then",
		"than",
		"which",
		"would",
		"shall",
		"should",
		"could",
		"about",
		"where",
		"here",
		"more",
		"most",
		"some",
		"such",
		"only",
		"very",
		"also",
		"just",
		"like",
		"upon",
		"through",
		"over",
		"under",
		"after",
		"before",
		"between",
		"his",
		"her",
		"its",
		"our",
		"not",
		"but",
		"for",
		"are",
		"was",
		"all",
		"can",
		"had",
		"has",
		"him",
		"she",
		"you",
		"who",
		"how",
		"why",
		"out",
		"now",
		"too",
		"any",
		"one",
		"two",
		"yet",
		"still",
		"macbeth",
		"lady",
		"duncan",
		"banquo",
		"macduff",
		"malcolm",
	}
)


class HfRow(TypedDict):
	node_type: str
	node_id: str
	act_id: str | None
	scene_id: str | None
	section_id: str | None
	speaker: str | None
	text: str | None
	motif_id: str
	motif_descriptor: str | None


class CleanRow(TypedDict):
	node_type: str
	node_id: str
	line_index: int
	citation: str
	act_id: str
	scene_id: str
	section_id: str
	speaker: str
	speaker_name: str
	text: str
	motif_id: str
	motif_descriptor: str


class DroppedLocalPair(TypedDict):
	node_id: str
	citation: str
	motif_id: str
	matched_trigram: str
	motif_descriptor: str


class CleaningReport(TypedDict):
	source_hf_repo: str
	source_hf_revision: str
	local_window_lines: int
	raw_total_rows: int
	raw_line_rows: int
	raw_unique_lines: int
	raw_unique_motifs: int
	dropped_motif_counts: dict[str, int]
	dropped_motifs_total: int
	clean_motif_inventory_size: int
	dropped_local_quote_pairs: int
	clean_unique_query_lines: int
	clean_line_motif_pairs: int
	clean_non_lexical_pairs: int
	clean_non_lexical_ratio: float
	clean_multi_line_motif_pairs: int
	clean_dispersed_gt50_pairs: int
	clean_dispersed_gt50_ratio: float
	dropped_motifs: dict[str, dict[str, str]]
	dropped_local_pairs: list[DroppedLocalPair]


@dataclass(frozen=True)
class PlayLine:
	index: int
	node_id: str
	act: int
	scene: int
	line_num: int
	speaker_id: str
	speaker_name: str
	section_id: str
	text: str

	@property
	def citation(self) -> str:
		return f"{self.act}.{self.scene}.{self.line_num}"


def format_speaker(speaker_id: str) -> str:
	if speaker_id in SPEAKER_OVERRIDES:
		return SPEAKER_OVERRIDES[speaker_id]
	base = re.sub(r"_Mac$", "", speaker_id)
	base = re.sub(r"([a-z])([A-Z])", r"\1 \2", base)
	return base.replace(".", " ").replace("_", " ").upper()


def clean_folger_text(raw: str) -> str:
	text = " ".join(raw.split())
	# Collapse spaces inserted by Folger XML token serialization before punctuation.
	text = re.sub(r"\s+([,.;:?!\)\]\}])", r"\1", text)
	text = re.sub(r"([\(\[\{])\s+", r"\1", text)
	text = re.sub(r"\s+([\u2014—])", r"\1", text)
	text = re.sub(r"([\u2014—])\s+", r"\1", text)
	text = re.sub(r"([“‘])\s+", r"\1", text)
	text = re.sub(r"\s+([”])", r"\1", text)
	return text.strip()


def ftln_sort_key(node_id: str) -> tuple[int, str]:
	match = re.match(r"ftln-(\d+)([a-z]*)", node_id)
	if match is None:
		return (999_999, node_id)
	return (int(match.group(1)), str(match.group(2)))


def normalize_for_ngrams(text: str) -> list[str]:
	lowered = re.sub(r"[\u2018\u2019']", "'", text.lower())
	cleaned = re.sub(r"[^a-z0-9'\s]", " ", lowered)
	return cleaned.split()


def content_trigrams(text: str) -> set[str]:
	tokens = normalize_for_ngrams(text)
	grams: set[str] = set()
	for i in range(len(tokens) - 2):
		tri = tokens[i : i + 3]
		if sum(1 for tok in tri if tok not in STOPWORDS) >= 2:
			grams.add(" ".join(tri))
	return grams


def content_words(text: str) -> set[str]:
	return {
		tok
		for tok in normalize_for_ngrams(text)
		if len(tok) >= 3 and tok not in STOPWORDS
	}


def load_play_lines(graph_path: Path) -> list[PlayLine]:
	raw = json.loads(graph_path.read_text(encoding="utf-8"))
	line_to_section: dict[str, str] = {}
	section_to_speaker: dict[str, str] = {}
	for edge in raw["edges"]:
		et = edge.get("edge_type")
		if et == "in_section":
			line_to_section[str(edge["target"])] = str(edge["source"])
		elif et == "spoken_by":
			section_to_speaker[str(edge["source"])] = str(edge["target"])

	raw_lines = [n for n in raw["nodes"] if n.get("node_type") == "line"]
	raw_lines.sort(
		key=lambda n: (
			int(n["node_reference"][0]),
			int(n["node_reference"][1]),
			int(n["node_reference"][2]),
			ftln_sort_key(str(n["id"])),
		)
	)

	play_lines: list[PlayLine] = []
	for idx, node in enumerate(raw_lines):
		node_id = str(node["id"])
		ref = node["node_reference"]
		sec_id = line_to_section.get(node_id, "")
		spk_id = section_to_speaker.get(sec_id, "UNKNOWN")
		play_lines.append(
			PlayLine(
				index=idx,
				node_id=node_id,
				act=int(ref[0]),
				scene=int(ref[1]),
				line_num=int(ref[2]),
				speaker_id=spk_id,
				speaker_name=format_speaker(spk_id),
				section_id=sec_id,
				text=clean_folger_text(str(node.get("text") or "")),
			)
		)
	return play_lines


def render_play_txt(play_lines: list[PlayLine]) -> str:
	out: list[str] = [
		"THE TRAGEDY OF MACBETH",
		"by William Shakespeare (Folger Digital Edition; Through-Line Anchors)",
		"=" * 74,
		"Format: [Act.Scene.Line | ftln-XXXX] Line text",
		"",
	]
	prev_act: int | None = None
	prev_scene: tuple[int, int] | None = None
	prev_section: str | None = None

	for line in play_lines:
		if line.act != prev_act:
			if prev_act is not None:
				out.append("")
			out.append(f"ACT {ROMAN_ACT.get(line.act, str(line.act))}")
			out.append("-" * 40)
			prev_act = line.act
			prev_scene = None
			prev_section = None
		if (line.act, line.scene) != prev_scene:
			out.append("")
			out.append(f"Act {line.act}, Scene {line.scene}")
			out.append("")
			prev_scene = (line.act, line.scene)
			prev_section = None
		if line.section_id != prev_section:
			out.append(f"{line.speaker_name}")
			prev_section = line.section_id
		out.append(f"  [{line.citation} | {line.node_id}] {line.text}")

	return "\n".join(out).rstrip() + "\n"


def _typst_escape(text: str) -> str:
	escaped = text.replace("\\", "\\\\")
	for ch in ("#", "[", "]", "*", "_", "$", "<", ">", "@", "`", "~", "/"):
		escaped = escaped.replace(ch, f"\\{ch}")
	return escaped


def render_play_pdf(play_lines: list[PlayLine], pdf_path: Path) -> None:
	typst_bin = shutil.which("typst")
	if typst_bin is None:
		raise RuntimeError("typst binary not found in PATH; cannot build macbeth_full.pdf")

	parts: list[str] = [
		'#set document(title: "The Tragedy of Macbeth — MotifBench Full Text", author: "William Shakespeare")',
		'#set page(paper: "us-letter", margin: (x: 0.75in, y: 0.75in), numbering: "1 / 1")',
		'#set text(font: ("Times New Roman", "Georgia", "New Computer Modern"), size: 9.5pt)',
		"#set par(leading: 0.55em, spacing: 0.35em)",
		'#align(center)[#text(size: 16pt, weight: "bold")[THE TRAGEDY OF MACBETH] \\ #text(size: 9.5pt, style: "italic")[William Shakespeare — Folger Digital Edition (CC BY-NC 3.0) with Folger Through-Line Anchors]]',
		"#v(0.8em)",
	]
	prev_act: int | None = None
	prev_scene: tuple[int, int] | None = None
	prev_section: str | None = None

	for line in play_lines:
		if line.act != prev_act:
			if prev_act is not None:
				parts.append("#pagebreak(weak: true)")
			act_label = ROMAN_ACT.get(line.act, str(line.act))
			parts.append(
				f'#v(0.6em)\n#text(size: 13pt, weight: "bold")[ACT {act_label}]\n#line(length: 100%, stroke: 0.5pt + gray)'
			)
			prev_act = line.act
			prev_scene = None
			prev_section = None
		if (line.act, line.scene) != prev_scene:
			parts.append(
				f'#v(0.5em)\n#text(size: 11pt, weight: "bold")[Act {line.act}, Scene {line.scene}]'
			)
			prev_scene = (line.act, line.scene)
			prev_section = None
		if line.section_id != prev_section:
			parts.append(
				f'#v(0.35em)\n#text(size: 8.5pt, weight: "bold", fill: rgb("#333333"))[{_typst_escape(line.speaker_name)}]'
			)
			prev_section = line.section_id
		anchor = _typst_escape(f"[{line.citation} | {line.node_id}]")
		body = _typst_escape(line.text)
		parts.append(
			f'#block(inset: (left: 1em), spacing: 0.2em)[#text(font: ("Menlo", "Courier New"), size: 7.5pt, fill: rgb("#555555"))[{anchor}] {body}]'
		)

	with tempfile.TemporaryDirectory() as tmpdir:
		typ_file = Path(tmpdir) / "macbeth_full.typ"
		typ_file.write_text("\n".join(parts) + "\n", encoding="utf-8")
		subprocess.run(
			[typst_bin, "compile", str(typ_file), str(pdf_path)],
			check=True,
			capture_output=True,
			text=True,
		)


def load_hf_rows(hf_jsonl: Path | None) -> list[HfRow]:
	if hf_jsonl is not None and hf_jsonl.exists():
		raw_text = hf_jsonl.read_text(encoding="utf-8")
	else:
		LOG.info("Downloading MotifBench macbeth.jsonl from %s", HF_MACBETH_URL)
		with urllib.request.urlopen(HF_MACBETH_URL) as resp:
			raw_text = resp.read().decode("utf-8")
	return [json.loads(line) for line in raw_text.splitlines() if line.strip()]


def clean_dataset(
	play_lines: list[PlayLine],
	hf_rows: list[HfRow],
	local_window: int = 15,
) -> tuple[list[CleanRow], list[dict[str, str]], CleaningReport]:
	line_by_id = {line.node_id: line for line in play_lines}
	line_trigrams = [content_trigrams(line.text) for line in play_lines]

	all_motifs: dict[str, str] = {}
	for row in hf_rows:
		mid = row["motif_id"]
		desc = row["motif_descriptor"] or ""
		if mid not in all_motifs:
			all_motifs[mid] = desc

	dropped_motifs: dict[str, dict[str, str]] = {}
	for mid, desc in sorted(all_motifs.items()):
		stripped = desc.strip()
		if not stripped:
			dropped_motifs[mid] = {"reason": "empty", "descriptor": desc}
		elif CITE_RE.search(stripped):
			dropped_motifs[mid] = {"reason": "explicit_citation", "descriptor": desc}
		elif POINTER_RE.search(stripped) or PURE_QUOTE_RE.search(stripped):
			dropped_motifs[mid] = {"reason": "passage_pointer", "descriptor": desc}

	kept_motifs = {
		mid: desc for mid, desc in sorted(all_motifs.items()) if mid not in dropped_motifs
	}
	motif_trigrams = {mid: content_trigrams(desc) for mid, desc in kept_motifs.items()}

	clean_rows: list[CleanRow] = []
	dropped_local_pairs: list[DroppedLocalPair] = []
	seen_pairs: set[tuple[str, str]] = set()
	raw_line_rows = [r for r in hf_rows if r["node_type"] == "line"]

	for row in raw_line_rows:
		lid = row["node_id"]
		mid = row["motif_id"]
		if mid not in kept_motifs:
			continue
		if (lid, mid) in seen_pairs:
			continue
		seen_pairs.add((lid, mid))

		pline = line_by_id[lid]
		d3 = motif_trigrams[mid]
		leaked_gram: str | None = None
		if d3:
			lo = max(0, pline.index - local_window)
			hi = min(len(play_lines), pline.index + local_window + 1)
			for j in range(lo, hi):
				common = d3 & line_trigrams[j]
				if common:
					leaked_gram = min(common)
					break
		if leaked_gram is not None:
			dropped_local_pairs.append(
				{
					"node_id": lid,
					"citation": pline.citation,
					"motif_id": mid,
					"matched_trigram": leaked_gram,
					"motif_descriptor": kept_motifs[mid],
				}
			)
			continue

		clean_rows.append(
			{
				"node_type": "line",
				"node_id": lid,
				"line_index": pline.index,
				"citation": pline.citation,
				"act_id": f"act_{pline.act}",
				"scene_id": f"act_{pline.act}_scene_{pline.scene}",
				"section_id": pline.section_id,
				"speaker": pline.speaker_id,
				"speaker_name": pline.speaker_name,
				"text": row["text"] or pline.text,
				"motif_id": mid,
				"motif_descriptor": kept_motifs[mid],
			}
		)

	clean_rows.sort(
		key=lambda r: (
			r["line_index"],
			int(r["motif_id"].split("_")[1]),
		)
	)

	# Compute surviving quality metrics
	non_lexical = 0
	by_motif: dict[str, list[int]] = defaultdict(list)
	for r in clean_rows:
		lw = content_words(r["text"])
		dw = content_words(r["motif_descriptor"])
		if not (lw & dw):
			non_lexical += 1
		by_motif[r["motif_id"]].append(r["line_index"])

	multi_line_pairs = 0
	dispersed_50 = 0
	for indices in by_motif.values():
		if len(indices) < 2:
			continue
		for idx in indices:
			multi_line_pairs += 1
			nearest = min(abs(idx - other) for other in indices if other != idx)
			if nearest > 50:
				dispersed_50 += 1

	motifs_list = [
		{"motif_id": mid, "descriptor": kept_motifs[mid]}
		for mid in sorted(kept_motifs, key=lambda m: int(m.split("_")[1]))
	]

	unique_lines = sorted({r["node_id"] for r in clean_rows})
	report: CleaningReport = {
		"source_hf_repo": "Acture/MotifBench",
		"source_hf_revision": HF_REVISION,
		"local_window_lines": local_window,
		"raw_total_rows": len(hf_rows),
		"raw_line_rows": len(raw_line_rows),
		"raw_unique_lines": len({r["node_id"] for r in raw_line_rows}),
		"raw_unique_motifs": len(all_motifs),
		"dropped_motif_counts": dict(
			Counter(v["reason"] for v in dropped_motifs.values())
		),
		"dropped_motifs_total": len(dropped_motifs),
		"clean_motif_inventory_size": len(motifs_list),
		"dropped_local_quote_pairs": len(dropped_local_pairs),
		"clean_unique_query_lines": len(unique_lines),
		"clean_line_motif_pairs": len(clean_rows),
		"clean_non_lexical_pairs": non_lexical,
		"clean_non_lexical_ratio": round(non_lexical / len(clean_rows), 4),
		"clean_multi_line_motif_pairs": multi_line_pairs,
		"clean_dispersed_gt50_pairs": dispersed_50,
		"clean_dispersed_gt50_ratio": round(dispersed_50 / multi_line_pairs, 4)
		if multi_line_pairs
		else 0.0,
		"dropped_motifs": dropped_motifs,
		"dropped_local_pairs": dropped_local_pairs,
	}
	return clean_rows, motifs_list, report


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
	parser.add_argument("--hf-jsonl", type=Path, default=None)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--local-window", type=int, default=15)
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	args.output_dir.mkdir(parents=True, exist_ok=True)

	play_lines = load_play_lines(args.graph)
	LOG.info("Loaded %d lines of Macbeth from %s", len(play_lines), args.graph)

	txt_path = args.output_dir / "macbeth_full.txt"
	txt_path.write_text(render_play_txt(play_lines), encoding="utf-8")
	LOG.info("Wrote %s (%d bytes)", txt_path, txt_path.stat().st_size)

	pdf_path = args.output_dir / "macbeth_full.pdf"
	render_play_pdf(play_lines, pdf_path)
	LOG.info("Wrote %s (%d bytes)", pdf_path, pdf_path.stat().st_size)

	hf_rows = load_hf_rows(args.hf_jsonl)
	clean_rows, motifs_list, report = clean_dataset(
		play_lines, hf_rows, local_window=args.local_window
	)

	clean_jsonl = args.output_dir / "macbeth_clean.jsonl"
	clean_jsonl.write_text(
		"\n".join(json.dumps(r, ensure_ascii=False) for r in clean_rows) + "\n",
		encoding="utf-8",
	)
	motifs_json = args.output_dir / "motifs_clean.json"
	motifs_json.write_text(
		json.dumps(motifs_list, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	report_json = args.output_dir / "cleaning_report.json"
	report_json.write_text(
		json.dumps(report, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)

	LOG.info(
		"Exported cleaned MotifBench: %d motifs, %d query lines, %d pairs (%.1f%% non-lexical, %.1f%% >50L dispersed)",
		report["clean_motif_inventory_size"],
		report["clean_unique_query_lines"],
		report["clean_line_motif_pairs"],
		report["clean_non_lexical_ratio"] * 100,
		report["clean_dispersed_gt50_ratio"] * 100,
	)


if __name__ == "__main__":
	main()
