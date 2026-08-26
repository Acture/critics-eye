import json
import logging

from pathlib import Path
from typing import List

import numpy as np
import torch
import typer
from networkx.readwrite import json_graph

from critics_eye.core.models.documents import PrimaryParserType
from critics_eye.core.primary import load_primary_data
from critics_eye.utils.embedding import build_feature_matrix
from critics_eye.core.primary.closure import build_closure_graph
from critics_eye.utils.graph import (
	convert_nx_to_hetero_data,
	load_graph_from_json,
	rehydrate_graph_enums,
	save_features_and_order,
)
from critics_eye.utils.perturb import PERTURBATIONS, PerturbKind
from critics_eye.utils.progress import new_progress


primary_app = typer.Typer(
	name="primary",
	help="Primary data pipeline: to_graph -> build_feature -> to_hetero [-> to_closure | perturb]",
)


@primary_app.command("to_graph", help="Parse one or more documents and output a graph.")
def to_graph(
	parser_type: PrimaryParserType = typer.Argument(
		...,
		help="Type of parser to use.",
		case_sensitive=False,
	),
	output: Path = typer.Option(
		Path.cwd(),
		"--output",
		"-o",
		help="Output directory path.",
		writable=True,
	),
	path: List[Path] = typer.Argument(
		...,
		help="Input file path(s).",
		exists=True,
	),
):
	"""
	Parse one or more documents。
	"""
	typer.echo("🚀 Starting parser...")
	typer.echo("  - Command: parse")
	typer.echo(f"  - Document Paths: {[p.name for p in path]}")
	typer.echo(f"  - Parser Type: {parser_type}")
	typer.echo(f"  - Output Directory: {output}")
	output.mkdir(parents=True, exist_ok=True)

	with new_progress() as progress:
		task = progress.add_task("Parsing source documents", total=len(path))
		for p in path:
			out_path = output / f"{p.stem}.json"
			if out_path.exists():
				progress.update(
					task,
					advance=1,
					description=f"Skipping {p.name} (already exists)",
				)
				continue

			progress.update(task, description=f"Parsing {p.name}")
			doc_graph = load_primary_data(p, parser_type)
			graph_data = json_graph.node_link_data(doc_graph, edges="edges")
			progress.update(task, description=f"Saving {out_path.name}")
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump(graph_data, f, indent=4, ensure_ascii=False)
			progress.advance(task, 1)


@primary_app.command("build_feature", help="Compute BGE embeddings for graph nodes.")
def build_feature(
	input_graph_json: Path = typer.Argument(..., help="Input graph JSON file."),
	embedding_model_name=typer.Option(
		"BAAI/bge-large-en-v1.5", help="Embedding model name."
	),
	output_dir: Path = typer.Option(
		Path.cwd(),
		"--output",
		"-o",
		help="Output directory path.",
		writable=True,
	),
):
	typer.echo("🚀 Starting feature builder...")
	typer.echo("  - Command: build_feature")
	output_dir.mkdir(parents=True, exist_ok=True)

	with new_progress() as progress:
		task = progress.add_task("Building graph features", total=3)
		progress.update(task, description=f"Loading {input_graph_json.name}")
		G = load_graph_from_json(input_graph_json)
		progress.advance(task, 1)

		progress.update(task, description="Embedding graph nodes")
		node_list, feature_matrix = build_feature_matrix(G, embedding_model_name)
		progress.advance(task, 1)
		progress.update(task, description="Saving feature artifacts")
		logging.info(f"Feature matrix shape: {feature_matrix.shape}")
		filename = input_graph_json.stem
		save_features_and_order(
			node_list,
			feature_matrix,
			output_dir / f"{filename}_order.json",
			output_dir / f"{filename}_features.npy",
		)
		progress.advance(task, 1)


@primary_app.command(
	"to_hetero",
	help="Convert graph JSON + features + node order to PyG HeteroData .pt file.",
)
def to_hetero(
	graph_input_path: Path = typer.Argument(
		...,
		help="[输入] Path to the graph.json file (e.g., <work>.json)",
		exists=True,
	),
	features_input_path: Path = typer.Argument(
		..., help="[输入] Path to the features.npy file.", exists=True
	),
	node_order_input_path: Path = typer.Argument(
		..., help="[输入] Path to the node_order.json file.", exists=True
	),
	output_dir: Path = typer.Option(
		Path.cwd(), "-o", help="[输出] Dir to save the final PyG .pt file."
	),
):
	"""
	一个独立的函数, 只负责将 .npy 和 node list 转换为 .pt。
	它假设 `build_features` 命令已经运行完毕。
	"""
	typer.echo("🚀 Starting PyG HeteroData conversion...")
	output_dir.mkdir(parents=True, exist_ok=True)

	with new_progress() as progress:
		task = progress.add_task("Converting graph to HeteroData", total=5)

		progress.update(task, description=f"Loading {graph_input_path.name}")
		G_full = load_graph_from_json(graph_input_path)
		rehydrate_graph_enums(G_full)
		progress.advance(task, 1)

		progress.update(task, description=f"Loading {features_input_path.name}")
		feature_matrix_X = np.load(features_input_path)
		progress.advance(task, 1)

		progress.update(task, description=f"Loading {node_order_input_path.name}")
		with open(node_order_input_path, "r", encoding="utf-8") as f:
			node_order = json.load(f)
		progress.advance(task, 1)

		progress.update(task, description="Validating node alignment")
		if G_full.number_of_nodes() != len(node_order):
			typer.echo(
				f"[bold red]错误: 图节点数 ({G_full.number_of_nodes()}) 与 "
				f"节点顺序列表 ({len(node_order)}) 不匹配。[/bold red]"
			)
			raise typer.Exit(code=1)

		if len(node_order) != feature_matrix_X.shape[0]:
			typer.echo(
				f"[bold red]错误: 节点顺序列表 ({len(node_order)}) 与 "
				f"特征矩阵 ({feature_matrix_X.shape[0]}) 不匹配。[/bold red]"
			)
			raise typer.Exit(code=1)
		progress.advance(task, 1)

		progress.update(task, description="Converting and saving PyG data")
		data_to_save = convert_nx_to_hetero_data(G_full, feature_matrix_X, node_order)
		output_path = output_dir / f"{graph_input_path.stem}.pt"
		torch.save(data_to_save, output_path)
		progress.advance(task, 1)

	typer.echo("[bold green]PyG 数据转换完成！[/bold green]")


@primary_app.command(
	"to_closure",
	help="Convert a hierarchical .pt to a line-only closure-edge graph (Variant B).",
)
def to_closure(
	input_pt: Path = typer.Argument(
		...,
		help="Path to the hierarchical HeteroData .pt file.",
		exists=True,
	),
	output_pt: Path = typer.Argument(
		...,
		help="Path to save the closure graph .pt file.",
	),
):
	"""Convert Variant A (hierarchical) to Variant B (line-only closure edges)."""
	with new_progress() as progress:
		task = progress.add_task("Building closure graph", total=3)
		progress.update(task, description=f"Loading {input_pt.name}")
		data = torch.load(input_pt, weights_only=False)
		progress.advance(task, 1)

		progress.update(task, description="Computing closure edges")
		closure_data = build_closure_graph(data)
		progress.advance(task, 1)

		progress.update(task, description=f"Saving {output_pt.name}")
		output_pt.parent.mkdir(parents=True, exist_ok=True)
		torch.save(closure_data, output_pt)
		progress.advance(task, 1)

	typer.echo(f"[bold green]Closure graph saved to {output_pt}[/bold green]")


@primary_app.command(
	"perturb",
	help="Apply a structural counterfactual perturbation to a HeteroData .pt file.",
)
def perturb(
	input_pt: Path = typer.Argument(
		...,
		help="Path to the source HeteroData .pt file.",
		exists=True,
	),
	output_pt: Path = typer.Argument(
		...,
		help="Path to save the perturbed .pt file.",
	),
	kind: PerturbKind = typer.Option(
		...,
		"--kind",
		help="Perturbation kind: scenes (P1), speakers (P2), or tree (P4).",
		case_sensitive=False,
	),
	seed: int = typer.Option(
		0,
		"--seed",
		help="Integer seed for the RNG.",
	),
):
	"""Re-write a structural edge family into a degree-matched random surrogate."""
	with new_progress() as progress:
		task = progress.add_task("Perturbing graph", total=3)
		progress.update(task, description=f"Loading {input_pt.name}")
		data = torch.load(input_pt, weights_only=False)
		progress.advance(task, 1)

		progress.update(task, description=f"Applying perturbation ({kind.value})")
		fn = PERTURBATIONS[kind.value]
		perturbed = fn(data, seed=seed)
		progress.advance(task, 1)

		progress.update(task, description=f"Saving {output_pt.name}")
		output_pt.parent.mkdir(parents=True, exist_ok=True)
		torch.save(perturbed, output_pt)
		progress.advance(task, 1)

	typer.echo(
		f"[bold green]Perturbed graph (kind={kind.value}, seed={seed}) "
		f"saved to {output_pt}[/bold green]"
	)
