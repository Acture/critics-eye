from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from critics_eye.utils import pdf as pdf_utils


class FakePage:
	def __init__(
		self,
		*,
		text: str,
		fonts_count: int,
		image_count: int = 0,
	) -> None:
		self._text = text
		self._fonts_count = fonts_count
		self._image_count = image_count

	def get_text(self, mode: str = "text") -> str:
		assert mode == "text"
		return self._text

	def get_fonts(self) -> list[tuple[object, ...]]:
		return [tuple()] * self._fonts_count

	def get_images(self, *, full: bool = True) -> list[tuple[object, ...]]:
		assert full is True
		return [tuple()] * self._image_count


class FakeDoc:
	def __init__(self, pages: list[FakePage]) -> None:
		self._pages = pages
		self.page_count = len(pages)

	def __getitem__(self, index: int) -> FakePage:
		return self._pages[index]

	def __iter__(self) -> Iterator[FakePage]:
		return iter(self._pages)

	def close(self) -> None:
		return None


def _make_report(classification: pdf_utils.PdfTextability) -> pdf_utils.PdfTextabilityReport:
	return pdf_utils.PdfTextabilityReport(
		pdf_hash="pdf-hash",
		page_count=12,
		classification=classification,
		textful_page_ratio=1.0 if classification == "native" else 0.0,
		scanned_like_page_ratio=1.0 if classification == "scanned" else 0.0,
		sampled_pages=[],
	)


def test_inspect_pdf_textability_classifies_native(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pdf_path = tmp_path / "native.pdf"
	pdf_path.write_text("placeholder", encoding="utf-8")
	doc = FakeDoc(
		[
			FakePage(text="A" * 120, fonts_count=2),
			FakePage(text="B" * 95, fonts_count=1),
			FakePage(text="C" * 100, fonts_count=3),
		]
	)

	monkeypatch.setattr(pdf_utils, "get_pdf_hash", lambda _: "native-hash")
	monkeypatch.setattr(pdf_utils, "_get_cached_textability_report", lambda _: None)
	monkeypatch.setattr(pdf_utils, "_set_cached_textability_report", lambda _: None)
	monkeypatch.setattr(pdf_utils.fitz, "open", lambda _: doc)

	report = pdf_utils.inspect_pdf_textability(pdf_path)

	assert report.classification == "native"
	assert report.page_count == 3
	assert report.textful_page_ratio == 1.0


def test_inspect_pdf_textability_classifies_scanned(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pdf_path = tmp_path / "scanned.pdf"
	pdf_path.write_text("placeholder", encoding="utf-8")
	doc = FakeDoc(
		[
			FakePage(text="", fonts_count=0, image_count=1),
			FakePage(text=" ", fonts_count=0, image_count=1),
			FakePage(text="\n", fonts_count=0, image_count=1),
		]
	)

	monkeypatch.setattr(pdf_utils, "get_pdf_hash", lambda _: "scanned-hash")
	monkeypatch.setattr(pdf_utils, "_get_cached_textability_report", lambda _: None)
	monkeypatch.setattr(pdf_utils, "_set_cached_textability_report", lambda _: None)
	monkeypatch.setattr(pdf_utils.fitz, "open", lambda _: doc)

	report = pdf_utils.inspect_pdf_textability(pdf_path)

	assert report.classification == "scanned"
	assert report.page_count == 3
	assert report.scanned_like_page_ratio == 1.0


def test_inspect_pdf_textability_classifies_mixed(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pdf_path = tmp_path / "mixed.pdf"
	pdf_path.write_text("placeholder", encoding="utf-8")
	doc = FakeDoc(
		[
			FakePage(text="A" * 120, fonts_count=2),
			FakePage(text="", fonts_count=0, image_count=1),
			FakePage(text="B" * 110, fonts_count=1),
			FakePage(text="", fonts_count=0, image_count=1),
		]
	)

	monkeypatch.setattr(pdf_utils, "get_pdf_hash", lambda _: "mixed-hash")
	monkeypatch.setattr(pdf_utils, "_get_cached_textability_report", lambda _: None)
	monkeypatch.setattr(pdf_utils, "_set_cached_textability_report", lambda _: None)
	monkeypatch.setattr(pdf_utils.fitz, "open", lambda _: doc)

	report = pdf_utils.inspect_pdf_textability(pdf_path)

	assert report.classification == "mixed"
	assert report.page_count == 4
	assert report.textful_page_ratio == 0.5
	assert report.scanned_like_page_ratio == 0.5


def test_inspect_pdf_textability_uses_cached_report(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pdf_path = tmp_path / "cached.pdf"
	pdf_path.write_text("placeholder", encoding="utf-8")
	cached = _make_report("native")

	monkeypatch.setattr(pdf_utils, "get_pdf_hash", lambda _: "cached-hash")
	monkeypatch.setattr(pdf_utils, "_get_cached_textability_report", lambda _: cached)
	monkeypatch.setattr(
		pdf_utils.fitz,
		"open",
		lambda _: pytest.fail("fitz.open should not run when preflight cache exists"),
	)

	report = pdf_utils.inspect_pdf_textability(pdf_path)

	assert report is cached


def test_extract_text_from_pdf_auto_routes_native(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pdf_path = tmp_path / "native.pdf"
	pdf_path.write_text("placeholder", encoding="utf-8")

	monkeypatch.setattr(pdf_utils, "get_pdf_hash", lambda _: "native-hash")
	monkeypatch.setattr(
		pdf_utils,
		"inspect_pdf_textability",
		lambda _: _make_report("native"),
	)
	monkeypatch.setattr(
		pdf_utils,
		"_extract_native_text_from_pdf",
		lambda _: "native text",
	)
	monkeypatch.setattr(
		pdf_utils,
		"_extract_text_with_full_ocr",
		lambda *args, **kwargs: pytest.fail("full OCR route should not run"),
	)
	monkeypatch.setattr(
		pdf_utils,
		"_extract_text_with_page_fallback",
		lambda *args, **kwargs: pytest.fail("page fallback route should not run"),
	)

	text = pdf_utils.extract_text_from_pdf(pdf_path)

	assert text == "native text"


def test_extract_text_from_pdf_auto_routes_mixed_to_page_fallback(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pdf_path = tmp_path / "mixed.pdf"
	pdf_path.write_text("placeholder", encoding="utf-8")
	calls: list[tuple[Path, str, str]] = []

	monkeypatch.setattr(pdf_utils, "get_pdf_hash", lambda _: "mixed-hash")
	monkeypatch.setattr(
		pdf_utils,
		"inspect_pdf_textability",
		lambda _: _make_report("mixed"),
	)
	monkeypatch.setattr(
		pdf_utils,
		"_extract_native_text_from_pdf",
		lambda *args, **kwargs: pytest.fail("native route should not run"),
	)
	monkeypatch.setattr(
		pdf_utils,
		"_extract_text_with_full_ocr",
		lambda *args, **kwargs: pytest.fail("full OCR route should not run"),
	)

	def fake_page_fallback(path: Path, pdf_hash: str, *, language: str) -> str:
		calls.append((path, pdf_hash, language))
		return "mixed text"

	monkeypatch.setattr(
		pdf_utils,
		"_extract_text_with_page_fallback",
		fake_page_fallback,
	)

	text = pdf_utils.extract_text_from_pdf(
		pdf_path,
		ocr_mode="auto",
		ocr_language="eng",
	)

	assert text == "mixed text"
	assert calls == [(pdf_path, "mixed-hash", "eng")]


def test_extract_text_from_pdf_auto_scanned_prefers_tesseract_when_ocrmypdf_missing(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	pdf_path = tmp_path / "scanned.pdf"
	pdf_path.write_text("placeholder", encoding="utf-8")
	calls: list[tuple[Path, str, str, int]] = []

	monkeypatch.setattr(pdf_utils, "get_pdf_hash", lambda _: "scanned-hash")
	monkeypatch.setattr(
		pdf_utils,
		"inspect_pdf_textability",
		lambda _: _make_report("scanned"),
	)
	monkeypatch.setattr(pdf_utils.shutil, "which", lambda _: None)
	monkeypatch.setattr(
		pdf_utils,
		"_extract_text_with_ocrmypdf",
		lambda *args, **kwargs: pytest.fail("OCRmyPDF should not run without binary"),
	)

	def fake_tesseract(
		path: Path,
		pdf_hash: str,
		*,
		language: str,
		workers: int,
	) -> str:
		calls.append((path, pdf_hash, language, workers))
		return "ocr text"

	monkeypatch.setattr(
		pdf_utils,
		"_extract_text_with_tesseract_ocr",
		fake_tesseract,
	)

	text = pdf_utils.extract_text_from_pdf(
		pdf_path,
		ocr_mode="auto",
		ocr_backend="auto",
		ocr_language="eng",
		ocr_workers=8,
	)

	assert text == "ocr text"
	assert calls == [(pdf_path, "scanned-hash", "eng", 8)]
