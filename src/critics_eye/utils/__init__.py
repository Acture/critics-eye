from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

from .pdf import extract_text_from_pdf
from .text import split_into_sentences
import json

__all__ = [
	"extract_text_from_pdf",
	"split_into_sentences",
	"save_as_jsonl",
]


def save_as_jsonl(data: Sequence[Any], filepath: Path) -> None:
	"""Saves a sequence of dictionaries as a JSONL file."""
	with open(filepath, "w", encoding="utf-8") as f:
		for item in data:
			json_line = json.dumps(item, ensure_ascii=False, indent=4)
			f.write(json_line + "\n")


def load_from_jsonl(
	filepath: Path,
	serailize_callback: Callable[[str], Any] | None = None,
) -> list[Any]:
	"""Loads a JSONL file into a list of dictionaries."""
	with open(filepath, "r", encoding="utf-8") as f:
		return [
			serailize_callback(line) if serailize_callback else json.loads(line)
			for line in f
		]
