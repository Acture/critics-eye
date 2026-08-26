from typing import Counter, Literal

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx import MultiDiGraph
from rich import Console, box
from rich.panel import Panel
from rich.table import Table

from critics_eye.core.models import StructuralType


def analysis(
	G_full: MultiDiGraph,
	level: Literal["high_level", "full"] = "high_level",
	draw_plots: bool = False,
):
	"""
	分析一个图, 可选择不同的“视图” (level),
	并计算详细的图复杂度指标, 使用 rich 库美观地输出。
	"""

	# 1. 初始化 Rich Console
	console = Console()

	# --- 2. 过滤图 (你的逻辑, 完美) ---
	G: MultiDiGraph
	title: str

	match level:
		case "high_level":
			title = "--- [1] 高层语义图 (High-Level Graph) 分析 ---"
			types_to_exclude = {StructuralType.LINE}
			nodes_to_keep = [
				n
				for n, d in G_full.nodes(data=True)
				if d.get("node_type") not in types_to_exclude
			]
			G = G_full.subgraph(nodes_to_keep)
		case "full":
			title = "--- [2] 完整图 (Full) 分析 ---"
			G = G_full
		case _:
			raise ValueError(f"未知的分析级别: {level}")

	# --- 3. 打印主标题 ---
	console.print(Panel(title, style="bold bright_blue", expand=False))

	if G.number_of_nodes() == 0:
		console.print("[red]图为空，无可分析。[/red]")
		return

	# --- 4. 节点类型详细统计 (我们先做这个) ---
	node_types = [d.get("node_type", "MissingType") for n, d in G.nodes(data=True)]
	type_counts = Counter(node_types)

	type_table = Table(
		title="[bold magenta]节点类型详细统计[/bold magenta]",
		box=box.MINIMAL_HEAVY_HEAD,
		padding=(0, 2),
	)
	type_table.add_column("类型", style="cyan")
	type_table.add_column("个数", style="white", justify="right")

	total_nodes = 0
	for node_type, count in sorted(type_counts.items(), key=lambda item: str(item[0])):
		type_name = str(node_type)
		if node_type == "MissingType" or node_type is None:
			type_table.add_row(
				f"[yellow]{type_name}[/yellow]", f"[yellow]{count}[/yellow]"
			)
		else:
			type_table.add_row(type_name, str(count))
		total_nodes += count

	type_table.add_row("[bold]总计[/bold]", f"[bold]{total_nodes}[/bold]")
	console.print(type_table)

	# --- 5. 异质性 (边类型) ---
	edge_types = [d.get("edge_type") for u, v, k, d in G.edges(keys=True, data=True)]
	console.print("\n[bold magenta]--- 边类型 (Unique) ---[/bold magenta]")
	console.print(set(edge_types))

	# --- 6. 创建一个主统计表 ---
	stats_table = Table(show_header=False, box=None, padding=(0, 2))
	stats_table.add_column(style="cyan", no_wrap=True)
	stats_table.add_column(style="white", justify="right")

	# --- 7. 基础统计 ---
	stats_table.add_row("[bold magenta]--- 基础统计 ---[/bold magenta]", "")
	stats_table.add_row("节点 (Nodes)", str(G.number_of_nodes()))
	stats_table.add_row("边 (Edges)", str(G.number_of_edges()))
	stats_table.add_row("图密度 (Density)", f"{nx.density(G):.6f}", end_section=True)

	# --- 8. 节点度分布 ---
	stats_table.add_row("[bold magenta]--- 节点度 (总) 分布 ---[/bold magenta]", "")
	degree_sequence = [d for n, d in G.degree()]
	if degree_sequence:
		stats_table.add_row("  平均度 (Mean)", f"{np.mean(degree_sequence):.2f}")
		stats_table.add_row("  中位数 (Median)", f"{np.median(degree_sequence):.0f}")
		stats_table.add_row("  最大度 (Max)", f"{np.max(degree_sequence)}")
		stats_table.add_row("  最小度 (Min)", f"{np.min(degree_sequence)}")
		stats_table.add_row(
			"  95百分位 (95th)",
			f"{np.percentile(degree_sequence, 95):.0f}",
			end_section=True,
		)

	# --- 9. 连通分量 ---
	stats_table.add_row("[bold magenta]--- 连通分量大小分布 ---[/bold magenta]", "")
	G_simple_undirected = nx.Graph(G)
	components = list(nx.connected_components(G_simple_undirected))
	component_sizes = [len(c) for c in components]

	if component_sizes:
		stats_table.add_row("  分量总数 (Total Components)", f"{len(component_sizes)}")
		stats_table.add_row("  最大分量大小 (Largest)", f"{np.max(component_sizes)}")
		stats_table.add_row(
			"  平均分量大小 (Mean Size)", f"{np.mean(component_sizes):.2f}"
		)
		stats_table.add_row(
			"  (大于1个节点的分量数)",
			f"{sum(1 for s in component_sizes if s > 1)}",
			end_section=True,
		)

	# --- 10. 路径与直径 (针对最大分量) ---
	stats_table.add_row(
		"[bold magenta]--- 路径与直径 (针对最大分量) ---[/bold magenta]", ""
	)
	if components:
		largest_component = max(components, key=len)
		largest_subgraph = G_simple_undirected.subgraph(largest_component)
		stats_table.add_row(
			"  (正在分析最大的子图)", f"{len(largest_component)} 个节点"
		)

		try:
			avg_path = nx.average_shortest_path_length(largest_subgraph)
			stats_table.add_row("  平均最短路径 (Avg. Path)", f"{avg_path:.2f}")
		except Exception as e:
			stats_table.add_row(
				"  平均最短路径 (Avg. Path)", f"[yellow]无法计算: {e}[/yellow]"
			)

		try:
			diameter = nx.diameter(largest_subgraph)
			stats_table.add_row("  图直径 (Diameter)", f"{diameter}")
		except Exception as e:
			stats_table.add_row(
				"  图直径 (Diameter)", f"[yellow]无法计算: {e}[/yellow]"
			)

		avg_clustering_undir = nx.average_clustering(G_simple_undirected)
		stats_table.add_row(
			"  平均聚类系数 (Avg. Clustering)", f"{avg_clustering_undir:.4f}"
		)
	else:
		stats_table.add_row("  (没有连通分量)", "")

	# --- 11. 打印主统计表 ---
	console.print(stats_table)

	# --- 12. 绘制图形 (不变) ---
	if draw_plots:
		console.print("\n[italic]正在生成绘图...[/italic]")
		plt.figure(figsize=(10, 6))
		plt.hist(degree_sequence, bins=50, log=True)
		plt.title(f"节点度分布 - {level}")
		plt.xlabel("度 (Degree)")
		plt.ylabel("节点数量 (Log Scale)")
		plt.show()

		plt.figure(figsize=(10, 6))
		plt.hist(component_sizes, bins=50, log=True)
		plt.title(f"连通分量大小分布 - {level}")
		plt.xlabel("分量大小 (Size)")
		plt.ylabel("分量数量 (Log Scale)")
		plt.show()
