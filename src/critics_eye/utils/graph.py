import json
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from networkx import MultiDiGraph
from rich import box
from rich.console import Console
from rich.table import Table
from torch_geometric.data import HeteroData

from critics_eye.core.models import (
	EdgeType,
	KeyNodeType,
	NodeType,
	SemanticEdgeType,
	StructuralEdgeType,
	StructuralNodeType,
	TextualNodeType,
	ThematicNodeType,
)
from critics_eye.core.models.edge import RelationLabel, SocialEdgeType, ThematicEdgeType


def load_graph_from_json(path: Path) -> MultiDiGraph:
	"""从 node-link JSON 文件加载图。"""
	logging.info(f"正在从 [cyan]{path}[/cyan] 加载图...")
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	# 确保它被正确加载为 MultiDiGraph
	G = nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")

	logging.info(
		f"  [green]✓[/green] 图已加载: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边"
	)
	rehydrate_graph_enums(G)
	return G


STR_TO_NODE_TYPE = {
	**{e.value: e for e in StructuralNodeType},
	**{e.value: e for e in TextualNodeType},
	**{e.value: e for e in ThematicNodeType},
	**{e.value: e for e in KeyNodeType},
}
# 创建 'contain' -> HardEdgeType.CONTAIN 的映射
STR_TO_EDGE_TYPE = {
	**{e.value: e for e in StructuralEdgeType},
	**{e.value: e for e in SemanticEdgeType},
	**{e.value: e for e in ThematicEdgeType},
	**{e.value: e for e in SocialEdgeType},
	**{e.value: e for e in RelationLabel},
}


def rehydrate_graph_enums(G: MultiDiGraph):
	"""
	(关键步骤)
	将图属性中的字符串 (如 "act") 转换回 Enum 对象 (如 StructuralType.ACT)。
	build_feature_matrix 依赖于 Enum 对象。
	"""
	logging.info("正在“重新水化”图的 Enum 类型...")
	nodes_rehydrated = 0
	edges_rehydrated = 0

	# 重新水化节点
	for node_id, data in G.nodes(data=True):
		node_type_str = data.get("node_type")
		if node_type_str and isinstance(node_type_str, str):
			data["node_type"] = STR_TO_NODE_TYPE.get(node_type_str, node_type_str)
			nodes_rehydrated += 1

	# 重新水化边
	for u, v, k, data in G.edges(keys=True, data=True):
		edge_type_str = data.get("edge_type")
		if edge_type_str and isinstance(edge_type_str, str):
			data["edge_type"] = STR_TO_EDGE_TYPE.get(edge_type_str, edge_type_str)
			edges_rehydrated += 1

	logging.info(
		f"  [green]✓[/green] 重新水化: {nodes_rehydrated} 个节点, {edges_rehydrated} 条边"
	)


def save_features_and_order(
	node_order: list, feature_matrix: np.ndarray, order_path: Path, features_path: Path
):
	"""将特征矩阵 (npy) 和节点顺序 (json) 保存到磁盘。"""
	logging.info(f"正在保存节点特征到 [cyan]{features_path}[/cyan]...")

	# 1. 保存 Numpy 数组
	#    np.save 是保存大型数组的标准、高效的方式
	np.save(features_path, feature_matrix)

	# 2. 保存节点顺序 (这个 JSON 列表是必需的)
	logging.info(f"正在保存节点顺序到 [cyan]{order_path}[/cyan]...")
	with open(order_path, "w", encoding="utf-8") as f:
		json.dump(node_order, f, indent=2)

	logging.info("  [green]✓[/green] 特征 (X) 和顺序 (order) 已保存。")


def convert_nx_to_hetero_data(
	G: nx.MultiDiGraph, X: np.ndarray, node_order: list
) -> HeteroData:
	"""
	将你的 networkx 图 G 和特征矩阵 X 转换为 PyG 的 HeteroData 对象。
	"""
	logging.info("正在将 NetworkX 图转换为 PyG HeteroData 对象...")
	data = HeteroData()

	# 1. 创建从 "node_id" 到 "整数索引" 的映射
	node_to_idx = {node_id: i for i, node_id in enumerate(node_order)}

	# 2. 按类型分配节点特征 X
	X_by_type = defaultdict(list)
	text_by_type = defaultdict(list)  # node_type -> [text str]
	id_by_type = defaultdict(list)  # node_type -> [node_id]
	original_to_type_idx = {}

	for i, node_id in enumerate(node_order):
		node_attr = G.nodes[node_id]

		node_type_enum: NodeType = G.nodes[node_id].get("node_type")
		if not node_type_enum:
			continue

		node_type_str = node_type_enum.value
		if not node_type_str:
			continue

		type_idx = len(X_by_type[node_type_str])
		X_by_type[node_type_str].append(X[i])
		original_to_type_idx[i] = (node_type_str, type_idx)

		id_by_type[node_type_str].append(node_id)
		text = node_attr.get("text", "")
		text_by_type[node_type_str].append(text)

	for node_type_str, vectors in X_by_type.items():
		data[node_type_str].x = torch.tensor(np.stack(vectors), dtype=torch.float)
		data[node_type_str].node_id = id_by_type[node_type_str]
		data[node_type_str].text = text_by_type[node_type_str]

	# 3. 按类型分配边 (Edge Index)
	edge_indices_by_type = defaultdict(list)
	for u_id, v_id, k, data_edge in G.edges(keys=True, data=True):
		edge_type_enum: EdgeType = data_edge.get("edge_type")
		if not edge_type_enum:
			continue

		edge_type_str = (
			edge_type_enum if isinstance(edge_type_enum, str) else edge_type_enum.value
		)
		if not edge_type_str:
			continue

		u_orig_idx, v_orig_idx = node_to_idx.get(u_id), node_to_idx.get(v_id)
		if u_orig_idx is None or v_orig_idx is None:
			logging.info(f"Skipping edge due to missing node IDs: {u_id} -> {v_id}")
			continue

		u_type_str, u_type_idx = original_to_type_idx[u_orig_idx]
		v_type_str, v_type_idx = original_to_type_idx[v_orig_idx]

		edge_key = (u_type_str, edge_type_str, v_type_str)
		edge_indices_by_type[edge_key].append((u_type_idx, v_type_idx))

	for edge_key, edges in edge_indices_by_type.items():
		data[edge_key].edge_index = (
			torch.tensor(edges, dtype=torch.long).t().contiguous()
		)

	logging.info("  [green]✓[/green] HeteroData 创建完毕。")
	logging.info(data)
	return data


def debug_edges(
	G: MultiDiGraph,
	source_type: NodeType,
	edge_type: EdgeType,
	target_type: NodeType,
):
	"""
	一个调试助手, 用于查找并打印所有匹配 (源类型, 边类型, 目标类型) 模式的边。
	"""
	console = Console()

	# 创建一个漂亮的 rich 表格
	table = Table(
		title=f"🐞 调试: ({source_type.name} -> {edge_type.name} -> {target_type.name})",
		box=box.MINIMAL_HEAVY_HEAD,
		padding=(0, 1),
		show_lines=True,
	)
	table.add_column("源节点 (Source Node)", style="cyan")
	table.add_column("源节点文本 (Snippet)", style="white")
	table.add_column("目标节点 (Target Node)", style="cyan")
	table.add_column("目标节点文本 (Snippet)", style="white")

	count = 0

	# 遍历图中的每一条边
	for u, v, k, data in G.edges(keys=True, data=True):
		# 1. 检查边的类型是否匹配
		if data.get("edge_type") == edge_type:
			# 2. 获取源节点和目标节点的数据
			# (使用 .get(u, {}) 来防止节点不存在时崩溃)
			u_data = G.nodes.get(u, {})
			v_data = G.nodes.get(v, {})

			# 3. 检查源节点和目标节点的类型是否匹配
			if (
				u_data.get("node_type") == source_type
				and v_data.get("node_type") == target_type
			):
				count += 1

				# 4. (为了调试) 提取节点的文本
				u_text = u_data.get("text", u_data.get("inner_text", "")) or ""
				v_text = v_data.get("text", v_data.get("inner_text", "")) or ""

				u_snippet = (u_text[:40] + "...") if len(u_text) > 40 else u_text
				v_snippet = (v_text[:40] + "...") if len(v_text) > 40 else v_text

				table.add_row(u, u_snippet, v, v_snippet)

	# --- 打印报告 ---
	if count == 0:
		console.print(
			f"[green]✓ 未找到匹配 ({source_type.name} -> {edge_type.name} -> {target_type.name}) 的边。[/green]"
		)
	else:
		console.print(table)
		console.print(
			f"[bold red]错误: 找到了 {count} 条不合逻辑的边。请检查你的解析器！[/bold red]"
		)
