"""Citations are parsed by code, not by the model.

The model used to be asked for `[3, 3, 106]`, which made it responsible for
reading Roman numerals. On a corpus where half the citations carry a line
number it returned one for only 20% of them, so `III. iii. 106` was silently
coarsened to a scene-level reference and the line-level labels -- the entire
benchmark signal -- were thrown away. It now copies the string and this parser
converts it.
"""

import pytest

from critics_eye.core.secondary.relation.citation import parse_citation
from critics_eye.core.secondary.relation.model import LitAnalysis


@pytest.mark.parametrize(
	("text", "expected"),
	[
		# Spaced Roman: the house style of 19th-century criticism, and the form
		# the original compact-dot regex could not see at all.
		("III. iii. 106", (3, 3, 106)),
		("(V. ii. 7)", (5, 2, 7)),
		("i. 5. 15", (1, 5, 15)),
		("V. ii.", (5, 2, None)),
		("IV. i. 10–12", (4, 1, 10)),
		# A range cites its first line.
		("II. i. 220-25", (2, 1, 220)),
		# Compact Arabic, the form the pipeline was originally built around.
		("1.7.16", (1, 7, 16)),
		("3.4", (3, 4, None)),
		("(2.1.33)", (2, 1, 33)),
		("[5.1.35]", (5, 1, 35)),
		# Spelled out.
		("Act III, Scene 2", (3, 2, None)),
		("Act III. sc. iii.", (3, 3, None)),
		("Act I, Scene 7, line 16", (1, 7, 16)),
		("Act II", (2, None, None)),
		# Labels that are themselves valid Roman numerals: "sc" contains c=100,
		# "l"=50 and "ll"=100, so a naive parser reads them as act numbers.
		("sc. iii", (None, 3, None)),
		("l. 42", (None, None, 42)),
		("ll. 42-45", (None, None, 42)),
		# A play name starting with L must not be read as a line label.
		("Lear III. i. 5", (3, 1, 5)),
	],
)
def test_citations_are_parsed(text: str, expected: tuple[int | None, int | None, int | None]) -> None:
	assert parse_citation(text) == expected


@pytest.mark.parametrize(
	"text",
	[
		"",
		None,
		"e.g.",
		# Project Gutenberg licence numbering, which parses as act.scene.line and
		# was the single largest source of false passages in the old gate.
		"1.E.1",
		"www.gutenberg.org",
	],
)
def test_non_citations_are_rejected(text: str | None) -> None:
	assert parse_citation(text) == (None, None, None)


def _record(**overrides: object) -> dict[str, object]:
	base: dict[str, object] = {
		"source_element": "x",
		"target_element": "y",
		"relationship_type": ["recall"],
		"source_citation": None,
		"target_citation": None,
		"justification_quote": "q",
		"confidence": 0.9,
	}
	base.update(overrides)
	return base


def test_model_parses_verbatim_citation_strings() -> None:
	record = LitAnalysis.model_validate(_record(source_citation="II. i. 44", target_citation="V. ii."))
	assert record.source_citation == (2, 1, 44)
	assert record.target_citation == (5, 2, None)


def test_model_still_accepts_preexisting_tuple_citations() -> None:
	"""JSONL written before the change must keep loading, or every earlier
	extraction becomes unreadable."""
	record = LitAnalysis.model_validate(_record(source_citation=[3, 3, 106], target_citation=[None, None, None]))
	assert record.source_citation == (3, 3, 106)
	assert record.target_citation == (None, None, None)


def test_missing_citation_becomes_an_empty_reference() -> None:
	record = LitAnalysis.model_validate(_record(source_citation=None))
	assert record.source_citation == (None, None, None)
