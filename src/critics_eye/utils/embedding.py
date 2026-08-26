from critics_eye.core.models import KeyNodeType

import logging
import numpy as np
from networkx import MultiDiGraph
from langchain_huggingface import HuggingFaceEmbeddings

# 引入你定义好的 Enum
from critics_eye.core.models import (
	TextualNodeType,
	ThematicNodeType,
	StructuralEdgeType,
)
from critics_eye.utils.progress import iter_with_progress

model_name = "BAAI/bge-large-en-v1.5"


#
# model_name = "Qwen/Qwen3-Embedding-0.6B"
#
# model_name = "Qwen/Qwen3-Embedding-8B"


def _embedding_text(node_id: str, node_data: dict[object, object]) -> str:
	text = node_data.get("text")
	if isinstance(text, str) and text.strip():
		return text.strip()

	display_text = node_data.get("display_text")
	if isinstance(display_text, str) and display_text.strip():
		return display_text.strip()

	tag = node_data.get("tag")
	if isinstance(tag, str) and tag.strip():
		return f"{tag}: {node_id}"

	return node_id


def build_feature_matrix(
	G: MultiDiGraph, model_name: str = "BAAI/bge-large-en-v1.5"
) -> tuple[list[str], np.ndarray]:
	"""
	构建图的节点特征矩阵 X。

	逻辑更新：
	1. 叶子节点 (Line, Keyword, Theme, Character, Action) -> 调用 API 嵌入文本。
	2. 容器节点 (Scene, Act, Play, Section) -> 聚合其组成部分 (successors on `contains`) 的平均向量。
	"""

	hf_embed_client = HuggingFaceEmbeddings(
		model_name=model_name,
		show_progress=True,
	)

	try:
		dummy_vector = hf_embed_client.embed_query("test")
		EMBED_DIM = len(dummy_vector)
		logging.info(f"✅ 模型 {model_name} 的嵌入维度为: {EMBED_DIM}")
	except Exception as e:
		logging.error(f"❌ 无法获取模型维度，默认设为 1024 (可能导致错误): {e}")
		EMBED_DIM = 1024  # 最后的兜底

	embeddings_cache: dict[str, np.ndarray] = {}

	# --- 1. 定义哪些是“叶子”节点 (直接有文本内容的) ---
	# 根据我们之前的定义，这些节点有具体的 text 字段
	LEAF_TYPES = {
		TextualNodeType.LINE,
		ThematicNodeType.THEME,  # 主题本身的描述
		KeyNodeType.CHARACTER,  # 人物本身的描述
		KeyNodeType.KEYWORD,
		KeyNodeType.GROUP,
		KeyNodeType.FORIEGN,
		KeyNodeType.NAME,
		KeyNodeType.SOUND,
		KeyNodeType.TITLE,
	}

	leaf_nodes = []
	leaf_texts = []

	for node_id, data in G.nodes(data=True):
		if node_id is None or node_id == "":
			raise ValueError(f"Node ID is empty: {node_id!r}")
		node_type = data.get("node_type")
		if node_type is None or node_type == "":
			raise ValueError(f"Node {node_id} has no 'node_type' attribute: {data}")
		if node_type in LEAF_TYPES:
			leaf_nodes.append(node_id)
			leaf_texts.append(_embedding_text(node_id, data))

	logging.info(f"正在嵌入 {len(leaf_texts)} 个叶子节点...")
	leaf_vectors = hf_embed_client.embed_documents(leaf_texts)

	for node_id, vector in zip(leaf_nodes, leaf_vectors):
		embeddings_cache[node_id] = np.array(vector)

	logging.info("叶子节点嵌入完成。")

	nodes_in_progress = set()

	# --- 3. 递归构建“容器”节点 (Scene, Act) ---
	def get_or_build_embedding(node_id: str) -> np.ndarray:
		if node_id in embeddings_cache:
			return embeddings_cache[node_id]

		# --- 循环检测 ---
		if node_id in nodes_in_progress:
			_log_cycle_error(G, node_id, nodes_in_progress)
			raise RecursionError(f"Detected dependency cycle at {node_id}")

		nodes_in_progress.add(node_id)

		node_type = G.nodes[node_id].get("node_type")

		if not node_type:
			raise ValueError(
				f"节点 {node_id} 没有定义 'node_type' 属性，无法构建嵌入。"
			)

		if node_type in LEAF_TYPES:
			logging.info(f"正在构建叶子 {node_type} 节点 {node_id} 的嵌入...")
			text = _embedding_text(node_id, G.nodes[node_id])
			my_vector = np.array(hf_embed_client.embed_query(text), dtype=float)
		else:
			logging.info(f"正在构建 {node_type} 节点 {node_id} 的嵌入...")
			vec_from_children: list[np.ndarray] = []
			for neighbor in G.successors(node_id):
				edge_dict = G.get_edge_data(node_id, neighbor)
				is_valid_child = False
				for key, attr in edge_dict.items():
					edge_type = attr.get("edge_type")
					if edge_type == StructuralEdgeType.CONTAINS:
						is_valid_child = True

				if is_valid_child:
					child_vec = get_or_build_embedding(neighbor)
					# 严格过滤无效向量
					if not np.isnan(child_vec).any():
						vec_from_children.append(child_vec)
			vec_from_self: np.ndarray | None = None
			text = G.nodes[node_id].get("text", "")

			if isinstance(text, str) and text.strip():
				try:
					# 检查之前批量嵌入时是否已经缓存了这个节点的文本向量
					# 如果这个节点被归类为 LEAF_TYPES，它应该已经在 embeddings_cache 里了
					# 但为了保险，这里再尝试获取或生成
					raw_vec = hf_embed_client.embed_query(text)
					vec_np = np.array(raw_vec, dtype=float)
					if vec_np.shape == (EMBED_DIM,):
						vec_from_self = vec_np
				except Exception:
					# logging.warning(f"节点 {node_id} 自身文本嵌入失败: {e}")
					pass

			candidate_vectors: list[np.ndarray] = []
			if vec_from_children:
				child_matrix = np.stack(vec_from_children)
				candidate_vectors.append(child_matrix.mean(axis=0))
			if vec_from_self is not None and vec_from_self.shape == (EMBED_DIM,):
				candidate_vectors.append(vec_from_self)

			if candidate_vectors:
				my_vector = np.stack(candidate_vectors).mean(axis=0)
			else:
				my_vector = np.zeros(EMBED_DIM, dtype=float)

		embeddings_cache[node_id] = my_vector
		nodes_in_progress.remove(node_id)
		return my_vector

	# --- 4. 触发构建 ---
	logging.info("正在计算叶子节点嵌入...")

	for node_id in iter_with_progress(
		G.nodes(),
		description="Building node embeddings",
		total=G.number_of_nodes(),
	):
		get_or_build_embedding(node_id)

	# --- 5. 堆叠最终矩阵 ---
	node_order = list(G.nodes())

	logging.info("正在检查向量完整性...")

	# 假设第一个向量是正常的，用它作为标准维度
	# 如果第一个就是坏的，我们需要找到第一个正常的来做标准
	expected_shape = None
	for nid in node_order:
		v = embeddings_cache.get(nid)
		if v is not None and hasattr(v, "shape") and len(v.shape) == 1:
			expected_shape = v.shape
			break

	if expected_shape is None:
		logging.error("❌ 严重错误：找不到任何一个有效的向量！所有节点都坏了？")
	else:
		logging.info(f"标准向量维度应为: {expected_shape}")

	# 遍历检查所有节点
	bad_nodes_found = False

	for node_id in iter_with_progress(
		node_order,
		description="Validating embedding matrix",
		total=len(node_order),
		transient=True,
	):
		# 使用 repr() 包裹 ID，这样 "" 会显示为 '', None 会显示为 None
		# 这能让你看清到底是空串、空格还是None
		id_repr = repr(node_id)

		node_data = G.nodes[node_id]
		node_type = node_data.get("node_type", "UNKNOWN")

		# 打印更多来源信息，帮助定位是哪里创建的
		tag = node_data.get("tag", "NoTag")
		xml_ref = node_data.get("node_reference", "NoRef")

		vec = embeddings_cache.get(node_id)
		if vec is None:
			logging.error("💀 找到坏节点!")
			logging.error(f"    ID (repr): {id_repr}")
			logging.error(f"    Type: {node_type}")
			logging.error(f"    XML Tag: {tag}")
			logging.error(f"    XML Ref: {xml_ref}")
			bad_nodes_found = True
			continue

		# 检查坏节点
		if np.ndim(vec) == 0 or np.isnan(vec).any():
			logging.error("💀 找到坏节点!")
			logging.error(f"    ID (repr): {id_repr}")  # 看看这里打印出什么！
			logging.error(f"    Type: {node_type}")
			logging.error(f"    XML Tag: {tag}")
			logging.error(f"    XML Ref: {xml_ref}")
			bad_nodes_found = True

	if bad_nodes_found:
		raise ValueError(
			"检测到无效节点，已终止运行以防止 np.stack 报错。请查看上方日志。"
		)

	feature_matrix = np.stack([embeddings_cache[node_id] for node_id in node_order])

	logging.info(f"特征矩阵构建完成: {feature_matrix.shape}")
	return node_order, feature_matrix


def _log_cycle_error(G, node_id, stack):
	"""辅助函数：打印详细的循环错误日志"""
	logging.warning("\n" + "=" * 80)
	logging.warning(f"!!! CYCLIC DEPENDENCY DETECTED AT {node_id} !!!")
	logging.warning(f"Recursion Stack: {stack}")

	# 打印入边 (在新的逻辑里，入边代表子节点)
	logging.warning("\nIncoming Edges (Potential Children):")
	for u, v, data in G.in_edges(node_id, data=True):
		logging.warning(f"  <- {u} [{data.get('edge_type')}]")

	logging.warning("=" * 80)
