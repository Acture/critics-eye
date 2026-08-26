"""Read act and scene numbers out of German ordinal headings.

The German DraCor corpora come from TextGrid digitisations of reading editions,
which carry no `@n` on their `<div>`s -- the number exists only as a word in the
`<head>`: "Erster Aufzug", "Dritte Szene", "Einundzwanzigster Auftritt". Without
recovering it every act and scene parses with an empty node reference, so
nothing anchors and nothing can be aligned against the English text.

The stem table is built from the headings actually present in gerdracor and
gershdracor, so it includes 19th-century spellings ("Eilfter" for eleventh,
"Zehenter" for tenth, "Siebender" for seventh) that a modern table would miss.
"""

from __future__ import annotations

import re

_UMLAUTS = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "s"})

# Ordinal stems after normalisation and inflection stripping. Variants are the
# archaic forms observed in the corpora, not hypothetical ones.
_ORDINALS = {
	"erst": 1,
	"zweit": 2,
	"dritt": 3,
	"viert": 4,
	"funft": 5,
	"sechst": 6,
	"siebent": 7,
	"siebend": 7,
	"siebt": 7,
	"acht": 8,
	"neunt": 9,
	"zehnt": 10,
	"zehent": 10,
	"elft": 11,
	"eilft": 11,
	"zwolft": 12,
	"dreizehnt": 13,
	"vierzehnt": 14,
	"funfzehnt": 15,
	"sechzehnt": 16,
	"sechszehnt": 16,
	"siebzehnt": 17,
	"siebenzehnt": 17,
	"achtzehnt": 18,
	"neunzehnt": 19,
	"zwanzigst": 20,
	"dreissigst": 30,
	"vierzigst": 40,
}

# Unit words as they appear on the left of "und" in compounds such as
# "Einundzwanzigster": these are cardinals, not ordinals.
_UNITS = {
	"ein": 1,
	"zwei": 2,
	"drei": 3,
	"vier": 4,
	"funf": 5,
	"sechs": 6,
	"sieben": 7,
	"acht": 8,
	"neun": 9,
}

_TENS = {"zwanzigst": 20, "dreissigst": 30, "vierzigst": 40}

_INFLECTIONS = ("es", "er", "en", "em", "e", "s")

# "Erster Aufzug." and "Erste Szene" both occur, as do headings that continue
# into a stage description ("Zweiter Auftritt. Ein Zimmer.").
_HEAD_WORD = re.compile(r"[a-z]+")

# Woyzeck's manuscript drafts number scenes with digits after the noun --
# "[H1, Szene 1]" -- so the digit must be taken from after "Szene", not from the
# draft sigil in front of it.
_HEAD_DIGIT = re.compile(r"(?:aufzug|akt|szene|scene|auftritt)\s*(\d+)")


def _normalise(word: str) -> str:
	return word.lower().translate(_UMLAUTS)


def _strip_inflection(stem: str) -> str:
	for suffix in _INFLECTIONS:
		if stem.endswith(suffix) and stem[: -len(suffix)] in _ORDINALS:
			return stem[: -len(suffix)]
	return stem


def parse_german_ordinal(text: str | None) -> int | None:
	"""Return the number named by the leading ordinal word, or None.

	Returns None for headings that name no number -- a location ("Ein Zimmer.")
	or a relative position ("Letzter Auftritt") -- so the caller can fall back to
	the position in the document.
	"""
	if not text:
		return None
	normalised = _normalise(text)

	digits = _HEAD_DIGIT.search(normalised)
	if digits is not None:
		return int(digits.group(1))

	match = _HEAD_WORD.search(normalised)
	if match is None:
		return None
	word = match.group()

	direct = _strip_inflection(word)
	if direct in _ORDINALS:
		return _ORDINALS[direct]

	if "und" in word:
		unit_part, _, tens_part = word.partition("und")
		tens = _strip_inflection(tens_part)
		if unit_part in _UNITS and tens in _TENS:
			return _UNITS[unit_part] + _TENS[tens]

	return None
