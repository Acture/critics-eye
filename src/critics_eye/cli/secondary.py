import logging
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
import typer
from networkx.readwrite import json_graph
import json

from sklearn.cluster import AgglomerativeClustering
from critics_eye.core.models.edge import RelationLabel, ThematicEdgeType

from critics_eye.core.models import (
	DocumentNode,
	StructuralNodeType,
	ThematicNodeType,
)
from critics_eye.core.secondary.motif.model import (
	LabeledMotifCluster,
	ReviewMotifIndex,
	MotifCluster,
)
from critics_eye.core.secondary.relation import (
	DEFAULT_LLM_MODEL,
	LitAnalysis,
	TRAPIRelationExtractor,
	extract_relations_from_text,
	select_relevant_passages,
)
from critics_eye.utils import load_from_jsonl
from critics_eye.utils.label_motif import generate_labels
from critics_eye.utils.graph import load_graph_from_json
from critics_eye.utils.pdf import extract_text
from critics_eye.utils.progress import iter_with_progress, new_progress
from critics_eye.utils.text import compute_embeddings

secondary_app = typer.Typer(
	name="secondary",
	help="Criticism pipeline: extract_relations -> build_review_embeddings -> cluster_review_motifs -> inject_review_and_motif -> label_motifs -> inject_labels",
)


def _add_source_file(analysis: dict, input_file: Path) -> dict:
	analysis["source_file"] = input_file.stem
	return analysis


type GraphNodeMap = dict[tuple[int | None, ...], DocumentNode]


def _normalize_phrase(phrase: str) -> str:
	return phrase.strip().lower()


def _resolve_member_node_id(
	review: LitAnalysis,
	kind: Literal["source", "target"],
	graph_nodes_map: GraphNodeMap,
) -> str | None:
	citation = review.source_citation if kind == "source" else review.target_citation
	node = graph_nodes_map.get(citation)
	return None if node is None else node.node_id


def _resolve_review_endpoint_id(
	review: LitAnalysis,
	kind: Literal["source", "target"],
	graph_nodes_map: GraphNodeMap,
	phrase_to_motif: dict[str, MotifCluster],
) -> str | None:
	node_id = _resolve_member_node_id(review, kind, graph_nodes_map)
	if node_id is not None:
		return node_id

	phrase = review.source_element if kind == "source" else review.target_element
	motif = phrase_to_motif.get(_normalize_phrase(phrase))
	if motif is None:
		logging.warning(
			"Phrase not in motif index for %s endpoint: %r, skipping relation injection",
			kind,
			phrase,
		)
		return None
	return f"motif_{motif.motif_id}"


@secondary_app.command(
	name="extract_relations",
	help="Extract relations from a document.",
)
def extract_relations(
	input_files: List[Path] = typer.Argument(..., help="Input file path.", exists=True),
	target_works: List[str] = typer.Option(
		..., help="Related work to extract relations for."
	),
	aliases: List[str] = typer.Option(
		[],
		"--alias",
		help=(
			"Extra names (characters, places) that count as a mention of the target "
			"work when selecting passages. Used for passage selection only -- the "
			"extraction prompt and output filename still use --target-works."
		),
	),
	output_dir: Path = typer.Option(Path.cwd(), "-o", help="Output file path."),
	model_id: str = typer.Option(
		DEFAULT_LLM_MODEL, "--model", help="TRAPI model ID for extraction."
	),
	timeout_seconds: float = typer.Option(
		180.0,
		"--timeout-seconds",
		help="Per-request TRAPI timeout in seconds.",
	),
	timeout_retries: int = typer.Option(
		1,
		"--timeout-retries",
		help="How many times to retry a failed TRAPI request.",
	),
	pdf_ocr_mode: str = typer.Option(
		"auto",
		"--pdf-ocr-mode",
		help="PDF OCR mode: auto, native, full_ocr, or page_fallback.",
	),
	pdf_ocr_backend: str = typer.Option(
		"auto",
		"--pdf-ocr-backend",
		help="PDF OCR backend: auto, tesseract, or ocrmypdf.",
	),
	pdf_ocr_language: str = typer.Option(
		"eng",
		"--pdf-ocr-language",
		help="OCR language passed to the PDF OCR backend.",
	),
	pdf_ocr_workers: int = typer.Option(
		4,
		"--pdf-ocr-workers",
		help="Worker count for full-document PDF OCR.",
	),
):
	output_dir.mkdir(parents=True, exist_ok=True)
	extractor = TRAPIRelationExtractor(
		model_id=model_id,
		timeout_seconds=timeout_seconds,
		timeout_retries=timeout_retries,
	)
	try:
		with new_progress() as progress:
			file_task = progress.add_task(
				"Extracting relations from documents",
				total=len(input_files),
			)
			for input_file in input_files:
				target_work_str = ", ".join(target_works)
				output_file = output_dir / f"{input_file.stem} - {target_work_str}.jsonl"
				if output_file.exists() and output_file.stat().st_size > 0:
					progress.console.print(
						f"[dim]SKIP (done): {output_file.name}[/dim]"
					)
					progress.advance(file_task, 1)
					continue
				progress.update(
					file_task, description=f"OCR + chunking {input_file.name}"
				)
				text = extract_text(
					input_file,
					pdf_ocr_mode=pdf_ocr_mode,
					pdf_ocr_backend=pdf_ocr_backend,
					pdf_ocr_language=pdf_ocr_language,
					pdf_ocr_workers=pdf_ocr_workers,
				)
				relative_texts = select_relevant_passages(
					text, [*target_works, *aliases]
				)

				# ── passage-level resume via .jsonl.tmp ──
				# tmp 文件用 passage 分隔符记录进度，中断后恢复时跳过已完成的 passages
				tmp_file = output_file.with_suffix(".jsonl.tmp")
				passage_sep = "### PASSAGE_DONE ###\n"
				done_passages = 0
				if tmp_file.exists():
					with open(tmp_file, encoding="utf-8") as existing:
						done_passages = existing.read().count(passage_sep)
					if done_passages > 0:
						progress.console.print(
							f"[yellow]RESUME: {input_file.name} "
							f"({done_passages}/{len(relative_texts)} passages done)[/yellow]"
						)

				piece_task = progress.add_task(
					f"Passages in {input_file.name}",
					total=len(relative_texts),
					completed=done_passages,
				)
				with open(tmp_file, "a", encoding="utf-8") as f:
					for i, relative_text in enumerate(relative_texts):
						if i < done_passages:
							continue
						progress.update(
							file_task,
							description=f"Extracting from {input_file.name} [{i + 1}/{len(relative_texts)}]",
						)
						res: list[LitAnalysis] = extract_relations_from_text(
							relative_text,
							target_works,
							extractor=extractor,
						)
						json_lines = [
							json.dumps(
								_add_source_file(
									analysis.model_dump(mode="json"), input_file
								),
								ensure_ascii=False,
							)
							for analysis in res
						]
						if json_lines:
							f.write("\n".join(json_lines) + "\n")
						f.write(passage_sep)
						f.flush()
						progress.advance(piece_task, 1)

				# 完成：去掉分隔符，rename 为最终文件
				with open(tmp_file, encoding="utf-8") as f:
					clean = f.read().replace(passage_sep, "")
				with open(output_file, "w", encoding="utf-8") as f:
					f.write(clean)
				tmp_file.unlink()
				progress.update(piece_task, visible=False)
				progress.advance(file_task, 1)
	finally:
		extractor.close()


@secondary_app.command(
	"build_review_embeddings",
	help="Embed source/target elements from extracted relations.",
)
def build_review_embeddings(
	review_paths: List[Path] = typer.Argument(
		..., help="One or more review JSONL file paths.", exists=True, dir_okay=False
	),
	out_embedding: Path = typer.Option(
		None, "--oe", help="Output embedding .npy file.", exists=False, writable=True
	),
	out_index_map: Path = typer.Option(
		None, "--oi", help="Output index mapping JSON.", exists=False, writable=True
	),
	embedding_model_name: str = typer.Option(
		"BAAI/bge-large-en-v1.5", help="Embedding model name."
	),
):
	logging.info(f"Loading reviews from {len(review_paths)} file(s)...")
	reviews: list[LitAnalysis] = []
	with new_progress() as progress:
		task = progress.add_task("Building review embeddings", total=4)
		for p in iter_with_progress(
			review_paths,
			description="Loading review files",
			total=len(review_paths),
			transient=True,
		):
			reviews.extend(load_from_jsonl(p, LitAnalysis.model_validate_json))
		progress.advance(task, 1)
		logging.info(f"Loaded {len(reviews)} reviews total.")

		phrases: list[str] = []
		index_map: list[ReviewMotifIndex] = []
		progress.update(task, description="Flattening source/target phrases")
		for i, r in enumerate(
			iter_with_progress(
				reviews,
				description="Preparing phrase index",
				total=len(reviews),
				transient=True,
			)
		):
			src = r.source_element
			tgt = r.target_element
			phrases.append(src)
			index_map.append(
				ReviewMotifIndex(
					index=len(index_map), review_id=i, kind="source", phrase=src
				)
			)
			phrases.append(tgt)
			index_map.append(
				ReviewMotifIndex(
					index=len(index_map), review_id=i, kind="target", phrase=tgt
				)
			)
		progress.advance(task, 1)
		logging.info(f"Total phrases: {len(phrases)} (reviews={len(reviews)})")

		progress.update(task, description=f"Embedding {len(phrases)} phrases")
		embeddings = compute_embeddings(phrases, embedding_model_name)
		progress.advance(task, 1)

		progress.update(task, description="Saving embedding artifacts")
		if out_embedding is None:
			out_embedding = review_paths[0].with_name(
				f"{review_paths[0].stem}_embedding.npy"
			)
		np.save(out_embedding, embeddings)
		logging.info(f"Saved embeddings → {out_embedding}")

		if out_index_map is None:
			out_index_map = review_paths[0].with_name(
				f"{review_paths[0].stem}_index_map.json"
			)

		out_index_map.write_text(
			json.dumps(
				[m.model_dump() for m in index_map], indent=2, ensure_ascii=False
			),
			encoding="utf-8",
		)
		logging.info(f"Saved index map → {out_index_map}")
		progress.advance(task, 1)


@secondary_app.command(
	"cluster_review_motifs",
	help="Cluster review elements into motifs via agglomerative clustering.",
)
def cluster_review_motifs(
	index_map_path: Path = typer.Argument(
		...,
		help="index map JSON from build_review_embeddings",
		exists=True,
		dir_okay=False,
	),
	embedding_path: Path = typer.Argument(
		...,
		help="embedding .npy from build_review_embeddings",
		exists=True,
		dir_okay=False,
	),
	out_motifs: Path = typer.Option(
		None, "--out", help="output motifs JSON", writable=True
	),
	distance_threshold: float = typer.Option(
		0.30, help="Agglomerative distance threshold (1 - cosine sim)"
	),
):
	with new_progress() as progress:
		task = progress.add_task("Clustering review motifs", total=6)

		progress.update(task, description=f"Loading {index_map_path.name}")
		with index_map_path.open("r", encoding="utf-8") as f:
			index_map: list[ReviewMotifIndex] = [
				ReviewMotifIndex.model_validate(line) for line in json.load(f)
			]

		E_all = np.load(embedding_path)
		assert E_all.shape[0] == len(
			index_map
		), f"Embedding rows ({E_all.shape[0]}) != index_map entries ({len(index_map)})"
		progress.advance(task, 1)

		progress.update(task, description="Normalizing review phrases")
		records: list[ReviewMotifIndex] = []
		for rec in iter_with_progress(
			index_map,
			description="Normalizing phrase inventory",
			total=len(index_map),
			transient=True,
		):
			records.append(
				ReviewMotifIndex(
					index=rec.index,
					review_id=rec.review_id,
					kind=rec.kind,
					phrase=rec.phrase.lower(),
				)
			)
		progress.advance(task, 1)
		typer.echo(f"Total non-empty phrases: {len(records)}")

		progress.update(task, description="Deduplicating phrase vectors")
		phrase_to_indices: dict[str, list[int]] = {}
		phrase_to_meta: dict[str, list[ReviewMotifIndex]] = {}
		for rec in iter_with_progress(
			records,
			description="Grouping duplicate phrases",
			total=len(records),
			transient=True,
		):
			p = rec.phrase
			idx = rec.index
			phrase_to_indices.setdefault(p, []).append(idx)
			phrase_to_meta.setdefault(p, []).append(
				ReviewMotifIndex(
					index=rec.index,
					review_id=rec.review_id,
					kind=rec.kind,
					phrase=p,
				)
			)

		uniq_phrases: list[str] = []
		uniq_vecs: list[np.ndarray] = []
		uniq_meta: list[list[ReviewMotifIndex]] = []
		for p, idx_list in iter_with_progress(
			phrase_to_indices.items(),
			description="Averaging duplicate embeddings",
			total=len(phrase_to_indices),
			transient=True,
		):
			vecs = E_all[idx_list]
			uniq_phrases.append(p)
			uniq_vecs.append(vecs.mean(axis=0))
			uniq_meta.append(phrase_to_meta[p])

		V = np.stack(uniq_vecs, axis=0)
		N, D = V.shape
		progress.advance(task, 1)
		typer.echo(f"Unique phrases: {N}, embedding dim: {D}")

		progress.update(task, description="Building cosine distance matrix")
		norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
		Vn = V / norms
		S = np.clip(Vn @ Vn.T, -1.0, 1.0)
		Dmat = (1.0 - S).astype(np.float64)
		progress.advance(task, 1)

		progress.update(task, description="Running agglomerative clustering")
		clustering = AgglomerativeClustering(
			metric="precomputed",
			linkage="average",
			distance_threshold=distance_threshold,
			n_clusters=None,
		)
		labels = clustering.fit_predict(Dmat)
		cluster_dict: dict[int, list[int]] = {}
		for idx, lab in enumerate(labels):
			cluster_dict.setdefault(int(lab), []).append(idx)

		motifs: list[MotifCluster] = []
		for motif_id, member_indices in iter_with_progress(
			cluster_dict.items(),
			description="Materializing motif clusters",
			total=len(cluster_dict),
			transient=True,
		):
			member_vecs = Vn[member_indices]
			centroid = member_vecs.mean(axis=0, keepdims=True)
			sims = (member_vecs @ centroid.T).reshape(-1)
			rep_local = int(sims.argmax())
			rep_global = member_indices[rep_local]
			rep_phrase = uniq_phrases[rep_global]

			member_occurrences: list[ReviewMotifIndex] = []
			for gi in member_indices:
				member_occurrences.extend(uniq_meta[gi])
			rep_member_index = next(
				(
					i
					for i, occurrence in enumerate(member_occurrences)
					if _normalize_phrase(occurrence.phrase)
					== _normalize_phrase(rep_phrase)
				),
				0,
			)

			motifs.append(
				MotifCluster(
					motif_id=motif_id,
					members=member_occurrences,
					representative_index=rep_member_index,
					representative_phrase=rep_phrase,
				)
			)

		motifs.sort(key=lambda m: -m.size)
		progress.advance(task, 1)

		progress.update(task, description="Saving motif clusters")
		if out_motifs is None:
			out_motifs = index_map_path.with_name(f"{index_map_path.stem}_motifs.json")

		out_motifs.write_text(
			json.dumps([m.model_dump() for m in motifs], indent=4, ensure_ascii=False),
			encoding="utf-8",
		)
		progress.advance(task, 1)

	typer.echo(f"Saved {len(motifs)} motifs → {out_motifs}")


@secondary_app.command(
	"inject_review_and_motif",
	help="Inject motif nodes and thematic edges into the graph.",
)
def inject_review_and_motif(
	graph_json: Path = typer.Argument(..., help="Input graph JSON file."),
	review_json: Path = typer.Argument(..., help="Input review JSON file."),
	index_map_json: Path = typer.Argument(..., help="Input index map JSON file."),
	motifs_json: Path = typer.Argument(..., help="Input motifs JSON file."),
	output_json: Path = typer.Option(
		None, "-o", help="Output graph JSON. Defaults to <input>_with_motif.json"
	),
	play: str = typer.Option(
		None,
		"--play",
		help=(
			"Only attach lines for reviews carrying this play tag. Required "
			"whenever the review file merges several plays."
		),
	),
):
	with new_progress() as progress:
		task = progress.add_task("Injecting review motifs into graph", total=5)

		progress.update(task, description=f"Loading {graph_json.name}")
		G = load_graph_from_json(graph_json)
		reviews: list[LitAnalysis] = load_from_jsonl(
			review_json, LitAnalysis.model_validate_json
		)
		motif_clusters = MotifCluster.from_json(motifs_json.read_text())
		progress.advance(task, 1)

		progress.update(task, description="Indexing graph + review state")
		review_map = {i: r for i, r in enumerate(reviews)}
		phrase_to_motif: dict[str, MotifCluster] = {
			_normalize_phrase(m.phrase): motif
			for motif in motif_clusters
			for m in motif.members
		}
		# Insertion-ordered dicts, not sets: every play is injected with the same
		# motif_clusters list, so iterating in insertion order is what makes the
		# theme nodes land at the same index in every play's .pt. Sets gave each
		# play its own arbitrary order, and motif i in Mac was a different motif
		# from motif i in Ham -- silently misaligning any cross-play training.
		nodes_to_add: dict[str, DocumentNode] = {}
		edges_to_add: dict[
			tuple[str, ThematicEdgeType | str | RelationLabel, str], None
		] = {}
		skipped_self_relations = 0
		skipped_other_play = 0

		# A citation is only (act, scene, line), so a merged review file gives
		# every play the same coordinates: Macbeth's 1.1.3 resolves against
		# Hamlet's 1.1.3 node and attaches Macbeth's motif to a Hamlet line.
		# The motif *inventory* is deliberately shared, but a line may only be
		# linked by criticism written about its own play.
		#
		# Called directly rather than through Typer, `play` arrives as the
		# OptionInfo default, which is neither a play name nor None.
		wanted_play = play if isinstance(play, str) else None

		def belongs_here(review: LitAnalysis) -> bool:
			if wanted_play is None:
				return True
			return review.play == wanted_play

		if wanted_play is not None and not any(r.play for r in reviews):
			raise typer.BadParameter(
				f"--play {wanted_play} was given but no review carries a play "
				"tag; the merged review file predates the tag and would "
				"silently attach every play's criticism to this graph."
			)

		graph_nodes: list[DocumentNode] = [
			DocumentNode.model_validate(n) for _, n in G.nodes(data=True)
		]
		# A line and the stage direction / section spanning it can share one
		# (act, scene, line) reference, so a plain dict comprehension resolved
		# citations by insertion order and silently handed 45 line-level
		# citations to non-line nodes, costing 45 motifs their only link.
		# Line nodes win; anything else only fills a reference no line claims.
		graph_nodes_map: dict[tuple, DocumentNode] = {}
		for n in graph_nodes:
			if n.node_reference is None or n.node_reference == (None, None, None):
				continue
			incumbent = graph_nodes_map.get(n.node_reference)
			if incumbent is not None and incumbent.node_type == StructuralNodeType.LINE:
				continue
			graph_nodes_map[n.node_reference] = n
		progress.advance(task, 1)

		progress.update(task, description="Resolving motif nodes and edges")
		for motif in iter_with_progress(
			motif_clusters,
			description="Injecting motif clusters",
			total=len(motif_clusters),
			transient=True,
		):
			node = DocumentNode(
				node_id=f"motif_{motif.motif_id}",
				node_reference=None,
				node_type=ThematicNodeType.THEME,
				tag="motif",
				text=motif.representative_phrase,
				display_text=motif.representative_phrase,
			)
			nodes_to_add[node.node_id] = node

			for member in motif.members:
				review = review_map[member.review_id]
				if not belongs_here(review):
					skipped_other_play += 1
					continue
				member_node_id = _resolve_member_node_id(
					review, member.kind, graph_nodes_map
				)
				if member_node_id is not None:
					edges_to_add[
						(
							member_node_id,
							ThematicEdgeType.REFLECTS_THEME,
							node.node_id,
						)
					] = None

				source_node_id = _resolve_review_endpoint_id(
					review, "source", graph_nodes_map, phrase_to_motif
				)
				target_node_id = _resolve_review_endpoint_id(
					review, "target", graph_nodes_map, phrase_to_motif
				)
				if source_node_id is None or target_node_id is None:
					continue
				if source_node_id == target_node_id:
					skipped_self_relations += 1
					continue

				for relation in review.relationship_type:
					edges_to_add[(source_node_id, relation, target_node_id)] = None
		progress.advance(task, 1)

		progress.update(task, description="Applying graph mutations")
		logging.info(
			"Added %s nodes, %s edges. Skipped %s self-relation edge candidates, "
			"%s members belonging to another play.",
			len(nodes_to_add),
			len(edges_to_add),
			skipped_self_relations,
			skipped_other_play,
		)
		for node in nodes_to_add.values():
			G.add_node(node.node_id, **node.model_dump())
		for s, relation, t in edges_to_add:
			G.add_edge(s, t, edge_type=relation)
		progress.advance(task, 1)

		progress.update(task, description="Saving enriched graph JSON")
		if output_json is None:
			output_json = graph_json.with_name(f"{graph_json.stem}_with_motif.json")
		graph_data = json_graph.node_link_data(G, edges="edges")
		with open(output_json, "w", encoding="utf-8") as f:
			json.dump(graph_data, f, indent=4, ensure_ascii=False)
		progress.advance(task, 1)

	logging.info(f"Saved graph with motifs → {output_json}")


@secondary_app.command(
	"inject_labels", help="Merge human-annotated motif labels into the .pt file."
)
def inject_labels(
	original_pt: Path = typer.Argument(..., help="Original PT file."),
	label_json: Path = typer.Argument(..., help="JSON file with labels."),
	outpath: Path = typer.Argument(..., help="Output JSON file path."),
):
	logging.info(f"Loading original data from {original_pt}...")
	with new_progress() as progress:
		task = progress.add_task("Injecting motif labels", total=4)
		progress.update(task, description=f"Loading {original_pt.name}")
		data = torch.load(original_pt, weights_only=False)
		progress.advance(task, 1)

		progress.update(task, description=f"Loading {label_json.name}")
		labeled_motifs: list[LabeledMotifCluster] = LabeledMotifCluster.from_json(
			label_json.read_text()
		)
		labeled_map = {m.motif_id: m for m in labeled_motifs}
		progress.advance(task, 1)

		node_type = "theme"
		if node_type not in data.node_types:
			raise ValueError(
				f"Node type '{node_type}' not found in data. Available: {data.node_types}"
			)

		num_motifs = data[node_type].num_nodes
		theme_string_ids = data[node_type].node_id
		original_texts = data[node_type].text
		new_short_labels = []

		progress.update(task, description="Aligning labels to motif IDs")
		for i in iter_with_progress(
			range(num_motifs),
			description="Merging motif labels",
			total=num_motifs,
			transient=True,
		):
			target_id_str = theme_string_ids[i]
			motif_id = int(target_id_str.split("_")[-1])
			labeled_motif = labeled_map[motif_id]
			original_text = original_texts[i]

			assert (
				labeled_motif.representative_phrase == original_text
			), f"Mismatched phrase: {labeled_motif.representative_phrase} vs {original_text}"
			new_short_labels.append(labeled_motif.labels)
		data[node_type].labels = new_short_labels
		progress.advance(task, 1)

		progress.update(task, description=f"Saving {outpath.name}")
		torch.save(data, outpath)
		progress.advance(task, 1)

	# 验证打印
	print("=" * 40)
	print("Verification (First 3 Motifs):")
	for i in range(3):
		print(f"ID {i}:")
		print(f"  [Text] : {data[node_type].text[i]}")
		print(f"  [Labels] : {data[node_type].labels[i][:50]}...")
	print("=" * 40)
	print("Done! You can now use the new .pt file.")


@secondary_app.command(
	"label_motifs", help="Generate short labels for motif clusters via LLM."
)
def label_motifs(
	motif_json_path: Path = typer.Argument(..., help="Path to the motifs JSON file."),
	output_path: Path = typer.Argument(
		..., help="Path to save the labeled motifs JSON."
	),
	model_id: str = typer.Option(
		DEFAULT_LLM_MODEL, "--model", help="TRAPI model ID for label generation."
	),
):
	generate_labels(motif_json_path, output_path, model_id=model_id)
