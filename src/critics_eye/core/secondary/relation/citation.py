"""Parse a Shakespeare citation string into `(act, scene, line)`.

The model used to be asked for `[3, 3, 106]` directly, which made it responsible
for reading Roman numerals and doing arithmetic. It did both badly: against a
corpus where half of all citations carry a line number, a run returned line
numbers for only 20% of them, silently coarsening `III. iii. 106` to
`[3, 3, null]`. Copying a string is a much easier task than converting one, so
the model now copies the citation verbatim and this module converts it.

The accepted forms are the ones measured in the corpus by
`scripts/survey_citation_formats.py`, not a guess at what criticism looks like.
Spaced Roman (`III. iii. 106`) is the house style of 19th-century criticism and
is far more common than the compact `1.7.16` form the pipeline was built around.
"""

from __future__ import annotations

import re

Citation = tuple[int | None, int | None, int | None]

_EMPTY: Citation = (None, None, None)

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

# An act or scene number, Roman or Arabic. Roman is matched first so that "i"
# is read as 1 rather than being rejected.
_NUM = r"(?:[MDCLXVImdclxvi]+|\d+)"

# "220-25" and "220–25" cite a range; the citation is its first line.
_LINE = r"\d+(?:\s*[-–]\s*\d+)?"

_ACT_WORD = r"\b[Aa]ct(?![A-Za-z])\.?"
# "sc", "l" and "ll" are themselves valid Roman numerals, so a label must not be
# allowed to match as a number: `l. 42` is line 42, not act 50 scene 42.
_SCENE_WORD = r"\b(?:[Ss]cene|[Ss]c)(?![A-Za-z])\.?"
_LINE_WORD = r"\b(?:[Ll]ines?|[Ll]{1,2})(?![A-Za-z])\.?"

# Guards a bare number against being taken out of the middle of a word, which is
# what let `sc. iii` parse as act 100.
_START = r"(?<![A-Za-z])"

_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
	# "Act III, Scene 2, line 15" and its abbreviated variants.
	(
		rf"{_ACT_WORD}\s*({_NUM})\s*[,.]?\s*{_SCENE_WORD}\s*({_NUM})\s*[,.]?\s*(?:{_LINE_WORD}\s*)?({_LINE})",
		("act", "scene", "line"),
	),
	(rf"{_ACT_WORD}\s*({_NUM})\s*[,.]?\s*{_SCENE_WORD}\s*({_NUM})", ("act", "scene")),
	# Before the dotted forms, so the "l" of `l. 42` is read as a label.
	(rf"{_LINE_WORD}\s*({_LINE})", ("line",)),
	# "III. iii. 106", "i. 5. 15", "1.7.16", "(2.1.33)". The line is Arabic in
	# every observed instance, which is what keeps this from swallowing a
	# three-part act/scene form that does not exist.
	(rf"{_START}({_NUM})\s*\.\s*({_NUM})\s*\.\s*({_LINE})", ("act", "scene", "line")),
	(rf"{_START}({_NUM})\s*\.\s*({_NUM})\s*\.?(?!\s*\d)", ("act", "scene")),
	(rf"{_ACT_WORD}\s*({_NUM})", ("act",)),
	(rf"{_SCENE_WORD}\s*({_NUM})", ("scene",)),
)


def _roman_to_int(text: str) -> int | None:
	total = 0
	previous = 0
	for char in reversed(text.lower()):
		value = _ROMAN_VALUES.get(char)
		if value is None:
			return None
		total += -value if value < previous else value
		previous = max(previous, value)
	return total or None


def _to_int(token: str) -> int | None:
	token = token.strip()
	# A range cites its first line.
	token = re.split(r"[-–]", token)[0].strip()
	if token.isdigit():
		return int(token) or None
	return _roman_to_int(token)


def parse_citation(text: object) -> Citation:
	"""Read `(act, scene, line)` out of a citation as the critic wrote it.

	Returns `(None, None, None)` when no citation is present, and leaves any
	component the citation does not give as None -- "V. ii." really is a
	scene-level reference and must not be invented into a line-level one.
	"""
	if text is None:
		return _EMPTY
	if not isinstance(text, str):
		return _EMPTY

	cleaned = re.sub(r"\s+", " ", text).strip().strip("()[]{}<>")
	if not cleaned:
		return _EMPTY

	for pattern, fields in _PATTERNS:
		match = re.search(pattern, cleaned)
		if match is None:
			continue
		found = dict(zip(fields, match.groups()))
		act = _to_int(found["act"]) if "act" in found else None
		scene = _to_int(found["scene"]) if "scene" in found else None
		line = _to_int(found["line"]) if "line" in found else None
		if (act, scene, line) != _EMPTY:
			return (act, scene, line)
	return _EMPTY
