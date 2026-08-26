"""German ordinal headings are the only place act and scene numbers exist in the
TextGrid-sourced DraCor corpora, so the table has to cover the 19th-century
spellings those editions actually use."""

import pytest

from critics_eye.core.primary.parsing.tei_dracor._ordinals import parse_german_ordinal


@pytest.mark.parametrize(
	("heading", "expected"),
	[
		("Erster Aufzug", 1),
		("Fünfter Aufzug", 5),
		("Erster Akt.", 1),
		("Erste Szene", 1),
		("Dritte Szene", 3),
		("Siebenter Auftritt", 7),
		("Zehnter Auftritt", 10),
		("Zwölfte Szene", 12),
		("Vierzehnter Auftritt", 14),
		("Zwanzigster Auftritt", 20),
		# Compounds: the left half is a cardinal, not an ordinal.
		("Einundzwanzigster Auftritt", 21),
		("Dreiundzwanzigster Auftritt", 23),
		# Archaic spellings present in the corpora; a modern table misses these.
		("Eilfter Auftritt", 11),
		("Zehenter Auftritt", 10),
		("Siebender Auftritt", 7),
		("Funfzehnter Auftritt", 15),
		("Sechszehnter Auftritt", 16),
		# Woyzeck's drafts number with digits, and the sigil comes first.
		("[H1, Szene 4]", 4),
	],
)
def test_ordinal_headings_are_read(heading: str, expected: int) -> None:
	assert parse_german_ordinal(heading) == expected


@pytest.mark.parametrize(
	"heading",
	[
		"Ein Zimmer.",
		"Straße.",
		"Letzter Auftritt",
		"Der Culenburgische Palast. Wohnung des Herzogs von Alba.",
		"",
		None,
	],
)
def test_headings_without_a_number_return_none(heading: str | None) -> None:
	"""A place name or a relative position names no number; the caller falls back
	to the position in the document rather than inventing one."""
	assert parse_german_ordinal(heading) is None
