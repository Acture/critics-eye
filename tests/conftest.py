# tests/conftest.py
import bs4
import pytest
from pathlib import Path

from critics_eye.core.secondary.relation import LitAnalysis, LitAnalysisList
from critics_eye.utils import extract_text_from_pdf

# Define the base directories as constants
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "fixture"


@pytest.fixture(scope="session")
def data_dir() -> Path:
	"""Provides the path to the test data directory."""
	return TEST_DATA_DIR


@pytest.fixture(scope="session")
def primary_data_dir() -> Path:
	"""Provides the path to the test data directory."""
	return TEST_DATA_DIR / "primary"


@pytest.fixture(scope="session")
def secondary_data_dir() -> Path:
	"""Provides the path to the test data directory."""
	return TEST_DATA_DIR / "secondary"


@pytest.fixture(scope="session")
def mac_reviews(secondary_data_dir: Path) -> list[LitAnalysis]:
	"""Provides the path to the test data directory."""
	p = secondary_data_dir / "mac_reviews.jsonl"
	return LitAnalysisList.from_jsonline(p).root


@pytest.fixture(scope="session")
def secondary_pdfs(secondary_data_dir: Path) -> list[Path]:
	"""Provides the path to the test data directory."""
	pdfs = list(secondary_data_dir.glob("*.pdf"))
	assert pdfs, "no pdfs found in test data directory"
	return pdfs


@pytest.fixture(scope="session")
def primary_xmls(primary_data_dir: Path) -> list[Path]:
	"""Provides the path to the test data directory."""
	return list(primary_data_dir.glob("*.xml"))


@pytest.fixture(scope="session")
def secondary_pdf_texts(secondary_pdfs: list[Path]) -> dict[Path, str]:
	texts = {
		pdf: extract_text_from_pdf(pdf) for pdf in list(secondary_pdfs)
	}  # 强制物化
	assert texts, "no texts extracted from pdfs"
	return texts


@pytest.fixture(scope="session")
def Mac_xml(primary_xmls: list[Path]) -> Path:
	for xml in primary_xmls:
		if xml.stem == "Mac":
			return xml
	raise ValueError("no Mac xml found")


@pytest.fixture(scope="session")
def Mac_text(Mac_xml: Path) -> str:
	soup = bs4.BeautifulSoup(Mac_xml.read_text(encoding="utf-8"), "xml")
	root = soup.find("TEI")
	text = root.find("text")

	text_body = text.find("body")
	return text_body.get_text()
