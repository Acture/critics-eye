import logging
from collections import Counter
from pathlib import Path
from typing import cast

import bs4
from lxml import etree

from critics_eye.utils.progress import iter_with_progress


def perform_tag_census(directory_path: Path, *, show_progress: bool = True) -> Counter:
	"""Traverses all XML files in a directory and counts tag occurrences."""
	tag_counter = Counter()
	xml_files = sorted(directory_path.rglob("*.xml"))
	file_iterable = (
		iter_with_progress(
			xml_files,
			description="Scanning XML files",
			total=len(xml_files),
		)
		if show_progress
		else xml_files
	)
	for file_path in file_iterable:
		try:
			tree = etree.parse(str(file_path))
			for element in cast(list[object], tree.xpath(".//*")):
				if not isinstance(element, etree._Element):
					continue
				clean_tag = element.tag.split("}")[-1]
				tag_counter[clean_tag] += 1
		except etree.XMLSyntaxError as e:
			logging.error(f"Failed to parse {file_path.name}: {e}")
	return tag_counter


def get_text(path: Path) -> str:
	soup = bs4.BeautifulSoup(path.read_text(encoding="utf-8"), "xml")
	root = soup.find("TEI")
	if root is None:
		raise ValueError(f"Missing TEI root in {path}")
	text = root.find("text")
	if text is None:
		raise ValueError(f"Missing <text> node in {path}")

	text_body = text.find("body")
	if text_body is None:
		raise ValueError(f"Missing <body> node in {path}")
	return text_body.get_text()
