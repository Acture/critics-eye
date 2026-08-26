from enum import StrEnum, auto
from typing import Union


class TextualNodeType(StrEnum):
	# --- Modality A: Textual ---
	LINE = auto()  # 具体的台词 (Line Nodes)


class ThematicNodeType(StrEnum):
	# --- Modality B: Thematic (Latent Anchors) ---
	THEME = auto()  # 主题 (Theme Nodes)


class StructuralNodeType(StrEnum):
	# --- Modality C: Structural ---
	PLAY = auto()
	ACT = auto()
	SCENE = auto()
	SECTION = auto()
	STAGE_DIRECTION = auto()
	LINE = auto()


class KeyNodeType(StrEnum):
	CHARACTER = auto()
	GROUP = auto()

	SOUND = auto()
	TITLE = auto()
	FORIEGN = auto()
	NAME = auto()

	KEYWORD = auto()


type NodeType = Union[
	TextualNodeType, ThematicNodeType, StructuralNodeType, KeyNodeType
]
