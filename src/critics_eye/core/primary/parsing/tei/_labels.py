from __future__ import annotations

from typing import Iterable

import bs4


def _normalize_text(text: str | None) -> str | None:
	if text is None:
		return None
	parts = text.split()
	if not parts:
		return None
	return " ".join(parts)


def tag_text(tag: bs4.element.Tag | None) -> str | None:
	if tag is None:
		return None
	return _normalize_text(tag.get_text(separator=" ", strip=True))


def first_tag_text(
	tag: bs4.element.Tag, selectors: Iterable[str], *, recursive: bool = True
) -> str | None:
	for selector in selectors:
		match = tag.find(selector, recursive=recursive)
		if isinstance(match, bs4.element.Tag):
			text = tag_text(match)
			if text:
				return text
	return None


def fallback_display_text(node_id: str) -> str:
	return node_id


def excerpt_display_text(text: str | None, *, limit: int = 80) -> str | None:
	normalized = _normalize_text(text)
	if normalized is None:
		return None
	if len(normalized) <= limit:
		return normalized
	return normalized[: limit - 1].rstrip() + "…"


def ref_display_text(prefix: str, reference: tuple[int | None, ...]) -> str | None:
	parts = [str(part) for part in reference if part is not None]
	if not parts:
		return None
	return f"{prefix} {'/'.join(parts)}"

