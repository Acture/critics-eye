from pathlib import Path
from typing import Annotated

import torch
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch_geometric.data import HeteroData

from critics_eye.utils.color import lerp_color
from critics_eye.utils.progress import iter_with_progress, new_progress
from critics_eye.utils.xml import perform_tag_census

util_app = typer.Typer(
	name="utils", help="Utility commands for data inspection and validation."
)


@util_app.command(
	"tag_census", help="Perform a tag census on a directory of Folger XML files."
)
def census(
	directory: Annotated[
		Path,
		typer.Argument(
			exists=True,
			file_okay=False,
			resolve_path=True,
			help="Path to the directory containing Folger XML files.",
		),
	],
):
	typer.echo(f"Performing tag census on directory: {directory}")

	all_tags = perform_tag_census(directory, show_progress=True)

	typer.echo("\n--- Tag Census Results (Sorted by Frequency) ---")

	header = typer.style(
		f"{'Tag Name':<20} | {'Count'}", bold=True, fg=typer.colors.BLUE
	)
	typer.echo(header)
	typer.echo("-" * 30)

	for tag, count in all_tags.most_common():
		typer.echo(f"{tag:<20} | {count}")


@util_app.command(
	"count_params", help="Count the number of parameters in a model checkpoint."
)
def count_parameters(
	model_paths: list[Path] = typer.Argument(..., help="Path to the model files."),
):
	console = Console()

	table = Table(title="Critic's Eye Model Complexity Analysis", box=box.ROUNDED)

	table.add_column("Checkpoint Name", style="cyan", no_wrap=True)
	table.add_column("Topology", style="magenta")
	table.add_column("Line Encoder (M)", justify="right", style="dim")
	table.add_column("Motif Encoder (M)", justify="right", style="dim")
	table.add_column("Scorer (M)", justify="right", style="dim")
	table.add_column("Total (M)", justify="right", style="bold green")

	results = []

	start_color = (0, 255, 255)
	end_color = (255, 0, 255)

	for path in iter_with_progress(
		sorted(model_paths, key=lambda p: p.name),
		description="Counting checkpoint parameters",
		total=len(model_paths),
	):
		try:
			checkpoint = torch.load(path, map_location="cpu", weights_only=False)

			if "line_encoder" not in checkpoint:
				continue

			def get_count(key):
				if key in checkpoint and isinstance(checkpoint[key], dict):
					return sum(
						p.numel()
						for p in checkpoint[key].values()
						if torch.is_tensor(p)
					)
				return 0

			c_line = get_count("line_encoder")
			c_motif = get_count("motif_encoder")
			c_scorer = get_count("scorer")

			total = c_line + c_motif + c_scorer
			topo_kind = path.stem.split("_")[1]

			match topo_kind:
				case "seq":
					topo_type = "Seq (Line Only)"
					topo_style = "bold spring_green1"
				case "hetero":
					topo_type = "Hetero (Line + Metadata)"
					topo_style = "bold dodger_blue1"
				case "hierarchical":
					topo_type = "Hierarchical (Line + Structure)"
					topo_style = "bold orange1"
				case "full":
					topo_type = "Full (Line + Metadata + Structure)"
					topo_style = "bold magenta1"
				case _:
					topo_type = "Unknown"
					topo_style = "dim white"

			results.append(
				{
					"name": path.name,
					"topo": topo_type,
					"topo_style": topo_style,
					"line": c_line / 1e6,
					"motif": c_motif / 1e6,
					"scorer": c_scorer / 1e6,
					"total": total / 1e6,
				}
			)

		except Exception:
			console.print(
				f"[red]Skipping {path.name}: Not a valid model checkpoint[/red]"
			)

	if not results:
		console.print("[red]No valid models found.[/red]")
		return

	all_totals = [r["total"] for r in results]
	min_total = min(all_totals)
	max_total = max(all_totals)

	for res in results:
		color_style = lerp_color(
			res["total"], min_total, max_total, start_color, end_color
		)
		color_name = (
			color_style.color.name if color_style.color is not None else "white"
		)

		table.add_row(
			res["name"],
			f"[{res['topo_style']}]{res['topo']}[/{res['topo_style']}]",
			f"{res['line']:.2f}",
			f"{res['motif']:.2f}",
			f"{res['scorer']:.2f}",
			f"[{color_name}]{res['total']:.2f}[/{color_name}]",
		)
	console.print(table)


@util_app.command(
	"check_hetero_data", help="Check the completeness of a PyG HeteroData file."
)
def check_hetero_data(
	file_path: Path = typer.Argument(..., help="Path to the PyG HeteroData file."),
):
	console = Console()
	console.print(
		Panel(
			f"--- Checking: [cyan]{file_path.name}[/cyan] ---",
			expand=False,
			style="bold",
		)
	)

	with new_progress() as progress:
		task = progress.add_task("Checking HeteroData file", total=4)

		if not file_path.exists():
			console.print("[bold red]Error: File not found![/bold red]")
			console.print(f"Path: {file_path.resolve()}")
			return

		progress.update(task, description=f"Loading {file_path.name}")
		try:
			data = torch.load(file_path, weights_only=False)
		except Exception as e:
			console.print("[bold red]CRITICAL: File load failed![/bold red]")
			console.print(e)
			return
		progress.advance(task, 1)
		console.print("[green]OK[/green] File loaded.")

		if not isinstance(data, HeteroData):
			console.print(
				f"[bold red]Error: Not a HeteroData object! Got: {type(data)}[/bold red]"
			)
			return

		progress.update(task, description="Validating HeteroData schema")
		data.validate()
		progress.advance(task, 1)
		console.print("[green]OK[/green] HeteroData validated.")

		progress.update(task, description="Checking node tensors")
		node_table = Table(title="[bold magenta]Node Check[/bold magenta]")
		node_table.add_column("Node Type", style="cyan")
		node_table.add_column("Count", style="white", justify="right")
		node_table.add_column("X Shape", style="white")
		node_table.add_column("NaN/Inf?", style="red")

		total_nodes = 0
		has_nan_inf = False
		for node_type in iter_with_progress(
			data.node_types,
			description="Inspecting node stores",
			total=len(data.node_types),
			transient=True,
		):
			x = data[node_type].x
			count = data[node_type].num_nodes
			shape = tuple(x.shape)

			nan_check = torch.isnan(x).any()
			inf_check = torch.isinf(x).any()
			status_str = []
			if nan_check:
				status_str.append("NaN")
			if inf_check:
				status_str.append("Inf")
			status = " ".join(status_str)
			if status:
				has_nan_inf = True

			node_table.add_row(
				node_type,
				str(count),
				str(shape),
				status if status else "[green]OK[/green]",
			)
			total_nodes += count

		node_table.add_row("[bold]Total[/bold]", f"[bold]{total_nodes}[/bold]", "", "")
		progress.advance(task, 1)
		console.print(node_table)
		if has_nan_inf:
			console.print("[bold red]Error: NaN or Inf in node features![/bold red]")

		progress.update(task, description="Checking edge indices")
		edge_table = Table(title="[bold magenta]Edge Check[/bold magenta]")
		edge_table.add_column("Edge Type", style="cyan", no_wrap=True)
		edge_table.add_column("Count", style="white", justify="right")
		edge_table.add_column("Shape", style="white")
		edge_table.add_column("OOB?", style="red")

		total_edges = 0
		has_oob = False
		for edge_type in iter_with_progress(
			data.edge_types,
			description="Inspecting edge stores",
			total=len(data.edge_types),
			transient=True,
		):
			edge_index = data[edge_type].edge_index
			count = data[edge_type].num_edges
			shape = tuple(edge_index.shape)

			src_type, _, dst_type = edge_type
			max_src_idx = data[src_type].num_nodes - 1
			max_dst_idx = data[dst_type].num_nodes - 1

			status = ""
			if count > 0:
				if edge_index[0].max() > max_src_idx:
					status = f"Source OOB! (Max: {edge_index[0].max()}, should be <= {max_src_idx})"
					has_oob = True
				if edge_index[1].max() > max_dst_idx:
					status = f"Target OOB! (Max: {edge_index[1].max()}, should be <= {max_dst_idx})"
					has_oob = True

			edge_table.add_row(
				f"({edge_type[0]}, [dim red]{edge_type[1]}[/dim red], {edge_type[2]})",
				str(count),
				str(shape),
				status if status else "[green]OK[/green]",
			)
			total_edges += count

		edge_table.add_row("[bold]Total[/bold]", f"[bold]{total_edges}[/bold]", "", "")
		progress.advance(task, 1)

	console.print(edge_table)
	if has_oob:
		console.print("[bold red]Error: Edge index out of bounds![/bold red]")

	if not has_nan_inf and not has_oob and total_nodes > 0 and total_edges > 0:
		console.print(
			"\n[bold green]All checks passed. Data is ready for training.[/bold green]"
		)
	else:
		console.print("\n[bold red]Checks failed. Fix your data pipeline.[/bold red]")
