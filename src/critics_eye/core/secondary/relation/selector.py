from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_CONTEXT_WINDOW = 2

# Citation shapes are derived from an audit of the actual corpus, not guessed;
# see scripts/survey_citation_formats.py and output/citation_survey/. The headline
# finding: the compact form the pipeline was built around (`1.7.16`, `III.iii.106`)
# occurs 0 times in the Macbeth criticism and once in 15,881 Othello chunks, while
# the spaced-Roman form (`III. iii. 106`) that 19th-century criticism actually uses
# was never matched at all.
_ROMAN = r"[IVXLivxl]+"
_NUM = r"\d+"
_ACTLIKE = rf"(?:{_ROMAN}|{_NUM})"

_CITATION_ALTERNATIVES = (
	# III.iii.106 / 1.7.16 -- compact, rare but valid.
	rf"\b{_ACTLIKE}\.{_ACTLIKE}\.{_NUM}\b",
	# 3.4 -- act.scene with no line. Constrained to acts 1-5 because a bare
	# dotted pair otherwise matches decimals, page numbers and version strings.
	r"\b[1-5]\.\d{1,3}\b",
	# III. iii. 106 -- the dominant form in the corpus.
	rf"\b{_ACTLIKE}\.\s+{_ACTLIKE}\.\s+{_NUM}",
	# V. ii. -- act and scene only.
	rf"\b{_ROMAN}\.\s+{_ROMAN}\.",
	# (V. ii. 7) / [5.1.35] / {2,1}
	rf"[\(\[\{{]\s*{_ACTLIKE}[.,]\s*{_ACTLIKE}(?:[.,]\s*{_NUM})?\s*[\)\]\}}]",
	# Act III, Scene 2 / Act III. sc. iii.
	rf"\bAct\.?\s+{_ACTLIKE}\b[.,;\s]*(?:sc(?:ene)?\.?\s+{_ACTLIKE})?",
	# sc. iii -- scene alone, common in running commentary.
	rf"\bsc(?:ene)?\.?\s+{_ACTLIKE}\b",
	# l. 42 / ll. 42-45
	rf"\bll?\.\s*{_NUM}(?:\s*[-\u2013]\s*{_NUM})?\b",
)
_CITATION_PATTERN = re.compile(
	"|".join(f"(?:{alt})" for alt in _CITATION_ALTERNATIVES),
	re.IGNORECASE,
)

# Project Gutenberg licence sections are numbered `1.E.1`, `1.F.3`, `1.C`, and the
# boilerplate is appended to every public-domain source. The previous dot pattern
# read those as act.scene.line, so ~44% of the Othello chunks sent to the extractor
# and ~70% of the Macbeth ones were licence text.
_BOILERPLATE_PATTERN = re.compile(
	r"Project\s+Gutenberg|gutenberg\.org|"
	r"\b1\.[A-F]\b|PGLAF|Literary\s+Archive\s+Foundation",
	re.IGNORECASE,
)


def split_text_for_extraction(
	raw_text: str,
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
	if not raw_text.strip():
		return []

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
	return [piece for piece in text_splitter.split_text(raw_text) if piece.strip()]


def contains_shakespeare_citation(text: str) -> bool:
	if _BOILERPLATE_PATTERN.search(text):
		return False
	return bool(_CITATION_PATTERN.search(text))


def contains_target_work(text: str, target_works: list[str]) -> bool:
	if not target_works:
		return False

	target_pattern = re.compile(
		"|".join(re.escape(title) for title in target_works),
		re.IGNORECASE,
	)
	return bool(target_pattern.search(text))


def select_relevant_passages(
	raw_text: str,
	target_works: list[str],
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
	context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> list[str]:
	"""Select chunks worth sending to the extractor.

	A chunk qualifies when it carries a Shakespeare citation *and* the target work
	is named within `context_window` chunks of it. The window exists because
	chunking is overlap-free: a passage in the middle of a chapter on Othello
	routinely cites `III. iii. 106` without repeating the play's name, so a
	strict same-chunk test throws away most of a monograph.

	The previous implementation fell back to *every* cited chunk whenever the
	target work was never named, which inverted the intent -- a source with three
	weak mentions was cut to three chunks while a source with none was passed
	through whole. Sources that never name the target now yield nothing.
	"""
	if not target_works:
		raise ValueError("target_works must not be empty")

	pieces = split_text_for_extraction(
		raw_text,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
	if not pieces:
		return []

	mentions = [contains_target_work(piece, target_works) for piece in pieces]
	near_mention = [
		any(mentions[max(0, i - context_window) : i + context_window + 1])
		for i in range(len(pieces))
	]

	return [
		piece
		for i, piece in enumerate(pieces)
		if near_mention[i] and contains_shakespeare_citation(piece)
	]
