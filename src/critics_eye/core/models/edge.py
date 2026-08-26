# --- 3. 边模型 (定义硬边和软边) ---
from enum import StrEnum
from typing import Union


class StructuralEdgeType(StrEnum):
	CONTAINS = "contains"
	APPEARS_IN = "appears_in"
	SPOKEN_BY = "spoken_by"
	NEXT = "next"  # Line -> Line (Sequential)


class SemanticEdgeType(StrEnum):
	MENTIONS = "mentions"


class ThematicEdgeType(StrEnum):
	# --- 3. Thematic Initialization (Weak Priors) ---
	REFLECTS_THEME = "reflects_theme"


class SocialEdgeType(StrEnum):
	# --- 1. Kinship (亲属关系) ---
	KIN_OF = "kin_of"

	# --- 2. Hierarchy & Service (阶级与服务) ---
	SERVES = "serves"  # Seyton -> Macbeth, Soldier -> Captain

	# --- 3. Group Membership (群体归属) ---
	MEMBER_OF = "member_of"  # WITCHES.1 -> WITCHES (Group Node)

	# --- 4. Political/Military (同僚/盟友) ---
	ALLY_OF = "ally_of"  # Banquo <-> Macbeth (初始状态)
	ENEMY_OF = "enemy_of"  # Macduff <-> Macbeth (剧情中后期，可作为初始负边或动态边)

	RELATED_TO = "related_to"


class RelationLabel(StrEnum):
	"""
	定义模型 Head B 需要预测的文学关系 (The Prediction Targets)
	这对应你原来的 'SoftEdgeType'，但现在它们是分类任务的 Label，而不是图的物理边
	"""

	# --- High-Level Categories (建议分组以提高准确率) ---

	# Group 1: Argumentative (逻辑/论证)
	SUPPORT = "support"
	CONTRADICT = "contradict"
	ELABORATE = "elaborate"

	# Group 2: Narrative (叙事/因果)
	CAUSE = "cause"
	FORESHADOW = "foreshadow"
	RECALL = "recall"  # 呼应

	# Group 3: Rhetorical (修辞/文学性 - 你的核心关注点)
	PARALLEL = "parallel"  # 平行
	CONTRAST = "contrast"  # 对比 (Triadic Closure 最擅长捕获这个)
	IRONY = "irony"  # 讽刺
	SYMBOLIZE = (
		"symbolize"  # 注意：这通常是 Line -> Keyword/Theme 的关系，而非 Line-Line
	)

	# Group 4: None
	NO_RELATION = "no_relation"


type EdgeType = Union[
	StructuralEdgeType,
	SemanticEdgeType,
	ThematicEdgeType,
	SocialEdgeType,
	RelationLabel,
]
