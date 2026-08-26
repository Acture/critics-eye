from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import fitz
import pytesseract
from PIL import Image

OCR_CACHE_DIR = Path.home() / ".cache" / "critics_eye" / "ocr"
DEFAULT_OCR_WORKERS = 4
SMALL_PDF_PAGE_THRESHOLD = 24
TEXTFUL_PAGE_CHAR_THRESHOLD = 80
SCANNED_PAGE_CHAR_THRESHOLD = 20

type PdfTextability = Literal["native", "scanned", "mixed"]
type PdfOcrMode = Literal["auto", "native", "full_ocr", "page_fallback"]
type PdfOcrBackend = Literal["auto", "tesseract", "ocrmypdf"]


@dataclass(slots=True)
class PdfPageScanStats:
	page_num: int
	fonts_count: int
	native_text_chars: int
	image_count: int
	is_textful: bool
	is_scanned_like: bool


@dataclass(slots=True)
class PdfTextabilityReport:
	pdf_hash: str
	page_count: int
	classification: PdfTextability
	textful_page_ratio: float
	scanned_like_page_ratio: float
	sampled_pages: list[PdfPageScanStats]


def get_pdf_hash(pdf_path: Path) -> str:
	"""Calculates the SHA256 hash of a file."""
	h = hashlib.sha256()
	with open(pdf_path, "rb") as f:
		while b := f.read(8192):
			h.update(b)
	return h.hexdigest()


def _normalize_ocr_mode(ocr_mode: str) -> PdfOcrMode:
	normalized = ocr_mode.strip().lower().replace("-", "_")
	if normalized not in {"auto", "native", "full_ocr", "page_fallback"}:
		raise ValueError(f"Unsupported PDF OCR mode: {ocr_mode}")
	return cast(PdfOcrMode, normalized)


def _normalize_ocr_backend(ocr_backend: str) -> PdfOcrBackend:
	normalized = ocr_backend.strip().lower().replace("-", "_")
	if normalized not in {"auto", "tesseract", "ocrmypdf"}:
		raise ValueError(f"Unsupported PDF OCR backend: {ocr_backend}")
	return cast(PdfOcrBackend, normalized)


def _pdf_cache_dir(pdf_hash: str) -> Path:
	return OCR_CACHE_DIR / pdf_hash


def _preflight_cache_path(pdf_hash: str) -> Path:
	return _pdf_cache_dir(pdf_hash) / "preflight.json"


def _full_ocr_text_cache_path(
	pdf_hash: str,
	backend: PdfOcrBackend,
	language: str,
) -> Path:
	return _pdf_cache_dir(pdf_hash) / f"full_{backend}_{language}.txt"


def _searchable_pdf_cache_path(
	pdf_hash: str,
	backend: PdfOcrBackend,
	language: str,
) -> Path:
	return _pdf_cache_dir(pdf_hash) / f"searchable_{backend}_{language}.pdf"


def get_cached_ocr_page(pdf_hash: str, page_num: int) -> str | None:
	"""Tries to retrieve a cached OCR result."""
	cache_path = _pdf_cache_dir(pdf_hash) / f"page_{page_num:04d}.txt"
	if cache_path.exists():
		return cache_path.read_text(encoding="utf-8")
	return None


def set_cached_ocr_page(pdf_hash: str, page_num: int, text: str) -> None:
	"""Saves an OCR result to the cache."""
	cache_path = _pdf_cache_dir(pdf_hash) / f"page_{page_num:04d}.txt"
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	cache_path.write_text(text, encoding="utf-8")


def _get_cached_textability_report(pdf_hash: str) -> PdfTextabilityReport | None:
	cache_path = _preflight_cache_path(pdf_hash)
	if not cache_path.exists():
		return None
	data = json.loads(cache_path.read_text(encoding="utf-8"))
	return PdfTextabilityReport(
		pdf_hash=str(data["pdf_hash"]),
		page_count=int(data["page_count"]),
		classification=cast(PdfTextability, data["classification"]),
		textful_page_ratio=float(data["textful_page_ratio"]),
		scanned_like_page_ratio=float(data["scanned_like_page_ratio"]),
		sampled_pages=[
			PdfPageScanStats(
				page_num=int(item["page_num"]),
				fonts_count=int(item["fonts_count"]),
				native_text_chars=int(item["native_text_chars"]),
				image_count=int(item["image_count"]),
				is_textful=bool(item["is_textful"]),
				is_scanned_like=bool(item["is_scanned_like"]),
			)
			for item in data["sampled_pages"]
		],
	)


def _set_cached_textability_report(report: PdfTextabilityReport) -> None:
	cache_path = _preflight_cache_path(report.pdf_hash)
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	cache_path.write_text(
		json.dumps(asdict(report), indent=2, ensure_ascii=False),
		encoding="utf-8",
	)


def _sample_page_numbers(page_count: int) -> list[int]:
	if page_count <= SMALL_PDF_PAGE_THRESHOLD:
		return list(range(page_count))

	candidates = {
		0,
		1,
		2,
		page_count // 4 - 1,
		page_count // 4,
		page_count // 2 - 1,
		page_count // 2,
		(3 * page_count) // 4 - 1,
		(3 * page_count) // 4,
		page_count - 3,
		page_count - 2,
		page_count - 1,
	}
	return sorted(i for i in candidates if 0 <= i < page_count)


def _inspect_page(page_num: int, page: fitz.Page) -> PdfPageScanStats:
	text = page.get_text("text")
	native_text_chars = len(text.strip())
	fonts_count = len(page.get_fonts())
	image_count = len(page.get_images(full=True))
	return PdfPageScanStats(
		page_num=page_num,
		fonts_count=fonts_count,
		native_text_chars=native_text_chars,
		image_count=image_count,
		is_textful=native_text_chars >= TEXTFUL_PAGE_CHAR_THRESHOLD,
		is_scanned_like=(
			fonts_count == 0 and native_text_chars < SCANNED_PAGE_CHAR_THRESHOLD
		),
	)


def inspect_pdf_textability(
	pdf_path: Path,
	*,
	force_refresh: bool = False,
) -> PdfTextabilityReport:
	pdf_hash = get_pdf_hash(pdf_path)
	if not force_refresh:
		cached = _get_cached_textability_report(pdf_hash)
		if cached is not None:
			return cached

	doc = fitz.open(pdf_path)
	try:
		page_count = doc.page_count
		sampled_pages = [_inspect_page(i, doc[i]) for i in _sample_page_numbers(page_count)]
	finally:
		doc.close()

	textful_count = sum(1 for page in sampled_pages if page.is_textful)
	scanned_like_count = sum(1 for page in sampled_pages if page.is_scanned_like)
	total_samples = max(1, len(sampled_pages))
	textful_ratio = textful_count / total_samples
	scanned_like_ratio = scanned_like_count / total_samples

	if textful_ratio >= 0.8:
		classification: PdfTextability = "native"
	elif scanned_like_ratio >= 0.8:
		classification = "scanned"
	else:
		classification = "mixed"

	report = PdfTextabilityReport(
		pdf_hash=pdf_hash,
		page_count=page_count,
		classification=classification,
		textful_page_ratio=textful_ratio,
		scanned_like_page_ratio=scanned_like_ratio,
		sampled_pages=sampled_pages,
	)
	_set_cached_textability_report(report)
	logging.info(
		"PDF textability: %s (textful=%.2f, scanned_like=%.2f, sampled=%s) for %s",
		report.classification,
		report.textful_page_ratio,
		report.scanned_like_page_ratio,
		len(report.sampled_pages),
		pdf_path.name,
	)
	return report


def _extract_native_text_from_pdf(pdf_path: Path) -> str:
	doc = fitz.open(pdf_path)
	try:
		return "\n\n".join(
			page.get_text("text").replace("\n", " ").strip()
			for page in doc
		).strip()
	finally:
		doc.close()


def _ocr_page_from_pdf(pdf_path: Path, page_num: int, language: str) -> str:
	doc = fitz.open(pdf_path)
	try:
		return ocr_page(doc[page_num], language=language)
	finally:
		doc.close()


def _extract_text_with_page_fallback(
	pdf_path: Path,
	pdf_hash: str,
	*,
	language: str,
) -> str:
	doc = fitz.open(pdf_path)
	full_text_parts: list[str] = []
	try:
		for page_num in range(doc.page_count):
			page = doc[page_num]
			stats = _inspect_page(page_num, page)
			if stats.is_scanned_like:
				logging.info(
					"Page %s appears scanned within mixed PDF. Using OCR fallback.",
					page_num + 1,
				)
				text = get_cached_ocr_page(pdf_hash, page_num)
				if text is None:
					text = ocr_page(page, language=language)
					set_cached_ocr_page(pdf_hash, page_num, text)
			else:
				text = page.get_text("text").replace("\n", " ")
			full_text_parts.append(text.strip())
	finally:
		doc.close()
	return "\n\n".join(part for part in full_text_parts if part).strip()


def _extract_text_with_tesseract_ocr(
	pdf_path: Path,
	pdf_hash: str,
	*,
	language: str,
	workers: int,
) -> str:
	full_cache = _full_ocr_text_cache_path(pdf_hash, "tesseract", language)
	if full_cache.exists():
		logging.info("Using cached full-document Tesseract OCR for %s", pdf_path.name)
		return full_cache.read_text(encoding="utf-8")

	doc = fitz.open(pdf_path)
	try:
		page_count = doc.page_count
	finally:
		doc.close()

	page_texts: dict[int, str] = {}
	missing_pages: list[int] = []
	for page_num in range(page_count):
		cached = get_cached_ocr_page(pdf_hash, page_num)
		if cached is None:
			missing_pages.append(page_num)
		else:
			page_texts[page_num] = cached

	if missing_pages:
		logging.info(
			"Running full-document Tesseract OCR for %s pages of %s (workers=%s)...",
			len(missing_pages),
			pdf_path.name,
			workers,
		)
		completed = 0
		if workers <= 1:
			for page_num in missing_pages:
				text = _ocr_page_from_pdf(pdf_path, page_num, language)
				page_texts[page_num] = text
				set_cached_ocr_page(pdf_hash, page_num, text)
				completed += 1
				if completed == len(missing_pages) or completed % 10 == 0:
					logging.info(
						"Tesseract OCR progress for %s: %s/%s pages",
						pdf_path.name,
						completed,
						len(missing_pages),
					)
		else:
			with ThreadPoolExecutor(max_workers=workers) as executor:
				futures: dict[Future[str], int] = {
					executor.submit(_ocr_page_from_pdf, pdf_path, page_num, language): page_num
					for page_num in missing_pages
				}
				for future in as_completed(futures):
					page_num = futures[future]
					text = future.result()
					page_texts[page_num] = text
					set_cached_ocr_page(pdf_hash, page_num, text)
					completed += 1
					if completed == len(missing_pages) or completed % 10 == 0:
						logging.info(
							"Tesseract OCR progress for %s: %s/%s pages",
							pdf_path.name,
							completed,
							len(missing_pages),
						)

	ordered_text = "\n\n".join(
		page_texts.get(page_num, "").strip() for page_num in range(page_count)
	).strip()
	full_cache.parent.mkdir(parents=True, exist_ok=True)
	full_cache.write_text(ordered_text, encoding="utf-8")
	return ordered_text


def _run_ocrmypdf(
	pdf_path: Path,
	output_pdf: Path,
	*,
	language: str,
	workers: int,
) -> None:
	output_pdf.parent.mkdir(parents=True, exist_ok=True)
	command = [
		"ocrmypdf",
		"--skip-text",
		"--quiet",
		"--jobs",
		str(max(1, workers)),
		"-l",
		language,
		str(pdf_path),
		str(output_pdf),
	]
	result = subprocess.run(command, capture_output=True, text=True)
	if result.returncode != 0:
		raise SystemError(
			"ocrmypdf failed with exit code "
			f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
		)


def _extract_text_with_ocrmypdf(
	pdf_path: Path,
	pdf_hash: str,
	*,
	language: str,
	workers: int,
) -> str:
	full_cache = _full_ocr_text_cache_path(pdf_hash, "ocrmypdf", language)
	if full_cache.exists():
		logging.info("Using cached OCRmyPDF text for %s", pdf_path.name)
		return full_cache.read_text(encoding="utf-8")

	searchable_pdf = _searchable_pdf_cache_path(pdf_hash, "ocrmypdf", language)
	if not searchable_pdf.exists():
		logging.info(
			"Running OCRmyPDF on %s with %s worker(s)...",
			pdf_path.name,
			workers,
		)
		_run_ocrmypdf(
			pdf_path,
			searchable_pdf,
			language=language,
			workers=workers,
		)

	text = _extract_native_text_from_pdf(searchable_pdf)
	full_cache.parent.mkdir(parents=True, exist_ok=True)
	full_cache.write_text(text, encoding="utf-8")
	return text


def _resolve_full_ocr_backend(ocr_backend: PdfOcrBackend) -> PdfOcrBackend:
	if ocr_backend == "auto":
		return "ocrmypdf" if shutil.which("ocrmypdf") else "tesseract"
	if ocr_backend == "ocrmypdf" and shutil.which("ocrmypdf") is None:
		raise SystemError(
			"OCRmyPDF not found. Install it with 'brew install ocrmypdf' or use '--pdf-ocr-backend tesseract'."
		)
	return ocr_backend


def _extract_text_with_full_ocr(
	pdf_path: Path,
	pdf_hash: str,
	*,
	ocr_backend: PdfOcrBackend,
	language: str,
	workers: int,
) -> str:
	resolved_backend = _resolve_full_ocr_backend(ocr_backend)
	if resolved_backend == "ocrmypdf":
		return _extract_text_with_ocrmypdf(
			pdf_path,
			pdf_hash,
			language=language,
			workers=workers,
		)
	return _extract_text_with_tesseract_ocr(
		pdf_path,
		pdf_hash,
		language=language,
		workers=workers,
	)


def extract_text(
	p: Path,
	*,
	pdf_ocr_mode: str = "auto",
	pdf_ocr_backend: str = "auto",
	pdf_ocr_language: str = "eng",
	pdf_ocr_workers: int = DEFAULT_OCR_WORKERS,
) -> str:
	match p.suffix:
		case ".pdf":
			return extract_text_from_pdf(
				p,
				ocr_mode=pdf_ocr_mode,
				ocr_backend=pdf_ocr_backend,
				ocr_language=pdf_ocr_language,
				ocr_workers=pdf_ocr_workers,
			)
		case ".txt":
			return p.read_text(encoding="utf-8")
		case _:
			raise ValueError(f"Unsupported file type: {p.suffix}")


def extract_text_from_pdf(
	pdf_path: Path,
	*,
	ocr_mode: str = "auto",
	ocr_backend: str = "auto",
	ocr_language: str = "eng",
	ocr_workers: int = DEFAULT_OCR_WORKERS,
) -> str:
	"""
	Extract text from a PDF using document-level preflight and an OCR strategy.
	"""
	mode = _normalize_ocr_mode(ocr_mode)
	backend = _normalize_ocr_backend(ocr_backend)
	workers = max(1, ocr_workers)
	pdf_hash = get_pdf_hash(pdf_path)

	if mode == "native":
		logging.info("Using native PDF text extraction for %s", pdf_path.name)
		return _extract_native_text_from_pdf(pdf_path)

	if mode == "full_ocr":
		logging.info("Using forced full-document OCR for %s", pdf_path.name)
		return _extract_text_with_full_ocr(
			pdf_path,
			pdf_hash,
			ocr_backend=backend,
			language=ocr_language,
			workers=workers,
		)

	if mode == "page_fallback":
		logging.info("Using page-fallback OCR for %s", pdf_path.name)
		return _extract_text_with_page_fallback(
			pdf_path,
			pdf_hash,
			language=ocr_language,
		)

	report = inspect_pdf_textability(pdf_path)
	if report.classification == "native":
		logging.info("Auto OCR mode selected native extraction for %s", pdf_path.name)
		return _extract_native_text_from_pdf(pdf_path)
	if report.classification == "scanned":
		logging.info("Auto OCR mode selected full-document OCR for %s", pdf_path.name)
		return _extract_text_with_full_ocr(
			pdf_path,
			pdf_hash,
			ocr_backend=backend,
			language=ocr_language,
			workers=workers,
		)
	logging.info("Auto OCR mode selected page fallback for %s", pdf_path.name)
	return _extract_text_with_page_fallback(
		pdf_path,
		pdf_hash,
		language=ocr_language,
	)


def ocr_page(page: fitz.Page, language: str = "eng") -> str:
	"""Helper function to OCR a single fitz.Page object."""
	try:
		pix = page.get_pixmap(dpi=300)
		img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
		return pytesseract.image_to_string(img, lang=language)
	except pytesseract.TesseractNotFoundError as exc:
		raise SystemError(
			"Tesseract-OCR not found. Please install it using 'brew install tesseract'"
		) from exc
	except Exception:
		page_number = page.number
		logging.exception(
			"Error during OCR on page %s",
			1 if page_number is None else page_number + 1,
		)
		return ""
