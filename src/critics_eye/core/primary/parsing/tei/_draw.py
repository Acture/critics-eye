import logging
from collections import defaultdict

import networkx as nx
from matplotlib import pyplot as plt
from networkx import MultiDiGraph

from critics_eye.core.models import (
	DocumentNode,
	KeyNodeType,
	RelationLabel,
	StructuralEdgeType,
	StructuralNodeType,
	ThematicNodeType,
)

edge_style_map = {
	# === 组 1: 硬边 (结构) ===
	# 目标: 低调、统一、作为背景
	StructuralEdgeType.CONTAINS: {
		"color": "#b0b0b0",
		"style": "dashed",
		"arrowstyle": "-",
		"width": 0.5,
		"alpha": 0.4,
	},
	StructuralEdgeType.NEXT: {
		"color": "#cccccc",
		"style": "dotted",
		"arrowstyle": "-",
		"width": 0.5,
		"alpha": 0.4,
	},
	StructuralEdgeType.APPEARS_IN: {
		"color": "#b0b0b0",
		"style": "solid",
		"arrowstyle": "->",
		"width": 0.5,
		"alpha": 0.4,
	},
	# === 组 2: 软边 - 因果与叙事 (剧情流) ===
	# 目标: 强烈、有活力的颜色，表示"流动"
	RelationLabel.CAUSE: {
		"color": "#ff4136",
		"style": "solid",
		"arrowstyle": "-|>",
		"width": 0.5,
	},  # 亮红
	RelationLabel.FORESHADOW: {
		"color": "#ff851b",
		"style": "dashed",
		"arrowstyle": "-|>",
		"width": 0.5,
	},  # 橙色
	RelationLabel.RECALL: {
		"color": "#ff851b",
		"style": "dashed",
		"arrowstyle": "<-",
		"width": 0.5,
	},  # 反向箭头
	# === 组 3: 软边 - 修辞与主题 (概念) ===
	# 目标: 沉稳、有深度的颜色，表示"连接"
	RelationLabel.PARALLEL: {
		"color": "#b100cd",
		"style": "solid",
		"arrowstyle": "<->",
		"width": 0.5,
	},  # 紫色
	RelationLabel.CONTRAST: {
		"color": "#b100cd",
		"style": "dashed",
		"arrowstyle": "<->",
		"width": 0.5,
	},  # 紫色虚线
}

node_style_map = {
	# 结构 (冷色调, 尺寸递减)
	StructuralNodeType.PLAY: {
		"color": "#FFD700",
		"size": 1500,
		"shape": "s",
	},  # 金色, 方形
	StructuralNodeType.ACT: {
		"color": "#0048BA",
		"size": 1000,
		"shape": "o",
	},  # 宝蓝色, 圆形
	StructuralNodeType.SCENE: {"color": "#ADD8E6", "size": 700, "shape": "o"},  # 亮蓝色
	# 内容 (中性色, 尺寸小)
	StructuralNodeType.SECTION: {
		"color": "#E0E0E0",
		"size": 300,
		"shape": "o",
	},  # 浅灰色 (Speech)
	StructuralNodeType.LINE: {
		"color": "#FFFFFF",
		"size": 150,
		"shape": "o",
	},  # 白色 (需要边框)
	# 概念 (暖色/亮色)
	KeyNodeType.KEYWORD: {
		"color": "#2ECC40",
		"size": 600,
		"shape": "d",
	},  # 绿色, 菱形 (Person)
	ThematicNodeType.THEME: {
		"color": "#B100CD",
		"size": 600,
		"shape": "p",
	},  # 紫色, 五边形
}


def draw_local_view(G: MultiDiGraph, root_id: str, radius: int = 1):
	"""
	(重构版)
	使用 'spring_layout' 绘制一个节点的“局部毛线球”。
	它只画 root_id 和它 'radius' 距离内的邻居。

	Args:
		G: 完整的 MultiDiGraph (3427+ 节点)
		root_id: 你想查看的中心节点 (e.g., "line-1.5.40")
		radius: 画几层邻居 (1 = 只画直接邻居)
	"""

	if root_id not in G:
		logging.error(f"根节点 '{root_id}' 在图中不存在。绘图中止。")
		return

	# --- 1. 提取一个小的“局部”子图 ---
	# G.neighbors(root_id) 会返回所有方向的邻居
	neighbor_nodes = list(G.neighbors(root_id))
	nodes_to_draw = set(neighbor_nodes) | {root_id}  # 用 set 来去重

	# (更高级: 使用 nx.ego_graph 来获取 N 跳邻居)
	# G_local = nx.ego_graph(G, root_id, radius=radius)

	G_local = G.subgraph(nodes_to_draw)
	logging.info(f"正在绘制局部视图: {root_id} (节点: {G_local.number_of_nodes()})")

	# --- 2. 按类型分组 (不变) ---
	nodes_by_type = defaultdict(list)
	for node, data in G_local.nodes(data=True):
		nodes_by_type[data.get("node_type") or "_default"].append(node)

	edges_by_type = defaultdict(list)
	for u, v, k, data in G_local.edges(keys=True, data=True):
		edges_by_type[data.get("edge_type") or "_default"].append((u, v, k))

	# --- 3. 布局计算 (钉住根节点) ---
	pos_fixed = {root_id: (0, 0)}
	fixed_nodes = [root_id]

	pos = nx.spring_layout(
		G_local,  # <-- 在“局部图”上计算布局
		pos=pos_fixed,
		fixed=fixed_nodes,
		k=0.9,  # k 值大一点, 拉开
		iterations=100,
		seed=42,
	)

	# --- 4. 绘制 (不变) ---
	plt.figure(figsize=(16, 12))

	# ... (你的节点/标签/边 绘制代码, 和之前一样) ...
	# (确保它们都使用 G_local 和 pos)

	for node_type, nodelist in nodes_by_type.items():
		style = node_style_map.get(node_type)

		if style:
			nx.draw_networkx_nodes(
				G_local,
				pos,
				nodelist=nodelist,
				node_color=style["color"],
				node_size=style["size"],
				node_shape=style["shape"],
				# 为白色节点添加边框，使其可见
				edgecolors="#333333" if style["color"] == "#FFFFFF" else None,
			)
		else:
			# 绘制没有定义样式的默认节点
			nx.draw_networkx_nodes(
				G_local, pos, nodelist=nodelist, node_size=300, node_color="gray"
			)

	# --- 6b. 绘制标签 ---
	nx.draw_networkx_labels(G_local, pos, font_size=8)

	# --- 6c. 循环绘制每种类型的边 ---
	for edge_type, edgelist in edges_by_type.items():
		style = edge_style_map.get(edge_type)
		if style:
			nx.draw_networkx_edges(
				G_local,
				pos,
				edgelist=edgelist,
				edge_color=style["color"],
				style=style["style"],
				arrowstyle=style["arrowstyle"],
				width=style["width"],
				alpha=style["alpha"],
			)
		else:
			nx.draw_networkx_edges(G_local, pos, edgelist=edgelist, edge_color="black")

	plt.title(f"Local View: {root_id} (Radius: {radius})")
	plt.axis("off")
	plt.show()


def draw_high_level_view(G_full: MultiDiGraph, root_id: str):
	"""
	(新函数)
	使用 'dot' 层级布局, 绘制一个干净的、不交叉的“组织结构图”。
	它只画“高层”节点 (非 LINE, WORD, CHAR)。
	"""

	# --- 1. 过滤图, 只保留高层节点 ---
	high_level_types = {
		StructuralNodeType.PLAY,
		StructuralNodeType.ACT,
		StructuralNodeType.SCENE,
		StructuralNodeType.SECTION,
		KeyNodeType.KEYWORD,
		ThematicNodeType.THEME,
	}
	nodes_to_keep = [
		n
		for n, data in G_full.nodes(data=True)
		if data.get("node_type") in high_level_types
	]
	G_high = G_full.subgraph(nodes_to_keep)
	logging.info(f"正在绘制高层视图... (节点: {G_high.number_of_nodes()})")

	# --- 2. 按类型分组 (不变) ---
	nodes_by_type = defaultdict(list)
	for node, data in G_high.nodes(data=True):
		nodes_by_type[data.get("node_type") or "_default"].append(node)
	edges_by_type = defaultdict(list)
	for u, v, k, data in G_high.edges(keys=True, data=True):
		edges_by_type[data.get("edge_type") or "_default"].append((u, v, k))

	# --- 3. 布局计算 (使用 'dot' + '重标号') ---
	pos = None
	try:
		# 3a. 创建骨架 (只含 contains)
		skeleton_graph = nx.DiGraph()
		skeleton_graph.add_nodes_from(G_high.nodes())
		if StructuralEdgeType.CONTAINS in edges_by_type:
			contain_edges = [
				(u, v) for u, v, k in edges_by_type[StructuralEdgeType.CONTAINS]
			]
			skeleton_graph.add_edges_from(contain_edges)

		# 3b. 重标号 (防崩溃)
		H = nx.convert_node_labels_to_integers(
			skeleton_graph, label_attribute="original_label"
		)

		# 3c. 找到整数根节点
		root_int = None
		for i, data in H.nodes(data=True):
			if data["original_label"] == root_id:
				root_int = i
				break
		if root_int is None:
			root_int = list(nx.topological_sort(H))[0]

		# 3d. 计算 'dot' 布局
		pos_int = nx.drawing.nx_pydot.graphviz_layout(H, prog="dot", root=root_int)

		# 3e. 翻译回原始 ID
		pos = {
			H.nodes[i]["original_label"]: xy
			for i, xy in pos_int.items()
			if "original_label" in H.nodes[i]
		}

	except Exception as e:
		logging.warning(f"Graphviz 'dot' 布局失败: {e}. 回退到 'spring_layout'.")
		pos = nx.spring_layout(G_high, k=0.5, seed=42)

	# --- 4. 绘制 (不变) ---
	plt.figure(figsize=(20, 16))

	# 4a. 绘制节点
	for node_type, nodelist in nodes_by_type.items():
		style = node_style_map.get(node_type, {})
		nx.draw_networkx_nodes(
			G_high,
			pos,
			nodelist=nodelist,
			node_color=style.get("color", "gray"),
			node_size=style.get("size", 300),
			node_shape=style.get("shape", "o"),
			edgecolors="#333333" if style.get("color") == "#FFFFFF" else None,
		)

	# 4b. 绘制标签
	nx.draw_networkx_labels(G_high, pos, font_size=8)

	# 4c. 绘制边
	for edge_type, edgelist in edges_by_type.items():
		style = edge_style_map.get(edge_type, {})

		# contains 边画直线, 其他画弧线
		rad = 0
		if edge_type != StructuralEdgeType.CONTAINS:
			rad = 0.1

		nx.draw_networkx_edges(
			G_high,
			pos,
			edgelist=edgelist,
			edge_color=style.get("color", "black"),
			style=style.get("style", "solid"),
			arrowstyle=style.get("arrowstyle", "->"),
			width=style.get("width", 1.0),
			alpha=style.get("alpha", 1.0),
			connectionstyle=f"arc3,rad={rad}",
		)

	plt.title(f"High-Level Graph (Root: {root_id}, Layout: 'dot')")
	plt.axis("off")
	plt.show()


def draw(G: MultiDiGraph, root_node: DocumentNode):
	nodes_by_type = defaultdict(list)
	for node, data in G.nodes(data=True):
		node_type = data.get("node_type")
		if node_type:
			nodes_by_type[node_type].append(node)
		else:
			nodes_by_type["_default"].append(node)  # 备用

	edges_by_type = defaultdict(list)
	for u, v, k, data in G.edges(keys=True, data=True):
		edge_type = data.get("edge_type")
		if edge_type:
			edges_by_type[edge_type].append((u, v, k))
		else:
			edges_by_type["_default"].append((u, v, k))

	pos_fixed = {root_node.node_id: (0, 0)}
	fixed_nodes = [root_node.node_id]

	pos = nx.spring_layout(
		G, pos=pos_fixed, fixed=fixed_nodes, iterations=1000, k=10, scale=10
	)

	plt.figure(figsize=(16, 12))

	# --- 6a. 循环绘制每种类型的节点 ---
	for node_type, nodelist in nodes_by_type.items():
		style = node_style_map.get(node_type)

		if style:
			nx.draw_networkx_nodes(
				G,
				pos,
				nodelist=nodelist,
				node_color=style["color"],
				node_size=style["size"],
				node_shape=style["shape"],
				# 为白色节点添加边框，使其可见
				edgecolors="#333333" if style["color"] == "#FFFFFF" else None,
			)
		else:
			# 绘制没有定义样式的默认节点
			nx.draw_networkx_nodes(
				G, pos, nodelist=nodelist, node_size=300, node_color="gray"
			)

	# --- 6b. 绘制标签 ---
	nx.draw_networkx_labels(G, pos, font_size=8)

	# --- 6c. 循环绘制每种类型的边 ---
	for edge_type, edgelist in edges_by_type.items():
		style = edge_style_map.get(edge_type)
		if style:
			nx.draw_networkx_edges(
				G,
				pos,
				edgelist=edgelist,
				edge_color=style["color"],
				style=style["style"],
				arrowstyle=style["arrowstyle"],
				width=style["width"],
				alpha=style["alpha"],
			)
		else:
			nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color="black")

	plt.title("Graph Visualization by Node and Edge Type (MultiDiGraph)")
	plt.axis("off")
	plt.show()
