from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from typer.testing import CliRunner

from critics_eye.cli import secondary as secondary_cli
from critics_eye.core.models.edge import RelationLabel
from critics_eye.core.secondary.relation import LitAnalysis
from critics_eye.core.secondary.relation.model import LitAnalysisList
from critics_eye.core.secondary.relation.prompt import (
	ALLOWED_RELATION_LABELS,
	build_relation_system_prompt,
)
from critics_eye.core.secondary.relation.selector import (
	contains_shakespeare_citation,
	select_relevant_passages,
)
from critics_eye.core.secondary.relation.service import extract_relations_from_text


class FakeExtractor:
	def __init__(self, outputs: list[str]) -> None:
		self.outputs = outputs
		self.calls: list[Sequence[BaseMessage]] = []

	def invoke(self, messages: Sequence[BaseMessage]) -> str:
		self.calls.append(messages)
		return self.outputs[len(self.calls) - 1]

	def close(self) -> None:
		return None


def _valid_output(
	source_element: str = "Blood imagery",
	target_element: str = "Macbeth's vision of death",
) -> str:
	return json.dumps(
		[
			{
				"source_element": source_element,
				"target_element": target_element,
				"relationship_type": ["foreshadow"],
				"source_citation": [2, 1, 33],
				"target_citation": [3, 4, 50],
				"justification_quote": "Blood imagery at (2.1.33) foreshadows the later vision at (3.4.50).",
				"confidence": 0.95,
			}
		]
	)


def test_select_relevant_passages_requires_a_nearby_target_mention() -> None:
	raw_text = "\n\n".join(
		[
			"At 1.7.12 the critic says ambition foreshadows ruin.",
			"Macbeth is explicitly named at 3.4.50, where the critic links the banquet scene to blood imagery.",
		]
	)

	passages = select_relevant_passages(raw_text, ["Macbeth"], chunk_size=120)

	assert passages
	assert any("Macbeth" in passage for passage in passages)


def test_select_relevant_passages_admits_cited_neighbours_of_a_mention() -> None:
	"""Chunking is overlap-free, so a passage mid-discussion cites without renaming
	the play. Such chunks are kept when a mention falls inside the context window."""
	raw_text = "\n\n".join(
		[
			"Macbeth is the critic's subject throughout this chapter.",
			"The dagger at 2.1.33 prefigures the later blood imagery, says the critic.",
		]
	)

	passages = select_relevant_passages(
		raw_text, ["Macbeth"], chunk_size=120, context_window=2
	)

	assert any("2.1.33" in passage for passage in passages)


def test_select_relevant_passages_drops_sources_that_never_name_the_target() -> None:
	"""Regression: the old implementation returned *every* cited passage when the
	target work was never mentioned, so a Hamlet run silently ingested whole
	volumes about other plays."""
	raw_text = "\n\n".join(
		[
			"The critic ties blood imagery to 2.1.33 and later guilt.",
			"A separate paragraph has no citation and should be dropped.",
		]
	)

	assert select_relevant_passages(raw_text, ["Hamlet"], chunk_size=120) == []


def test_select_relevant_passages_matches_spaced_roman_citations() -> None:
	"""The dominant citation style in 19th-century criticism."""
	raw_text = (
		"Macbeth's vision at III. iv. 106 is read by the critic as the turning point."
	)

	assert select_relevant_passages(raw_text, ["Macbeth"], chunk_size=200)


def test_select_relevant_passages_rejects_gutenberg_boilerplate() -> None:
	"""Licence section numbers such as 1.E.1 parse as act.scene.line, so the
	boilerplate appended to every public-domain source used to dominate the input."""
	raw_text = (
		"Macbeth. Section 1.E.1 of the Project Gutenberg License applies to "
		"copying and distributing this work; see www.gutenberg.org for terms."
	)

	assert select_relevant_passages(raw_text, ["Macbeth"], chunk_size=400) == []


def test_contains_shakespeare_citation_detects_common_forms() -> None:
	assert contains_shakespeare_citation("The critic cites 1.7.12 directly.")
	assert contains_shakespeare_citation("The critic cites (Act IV).")
	assert contains_shakespeare_citation("The critic cites III. iv. 106 here.")
	assert contains_shakespeare_citation("Compare V. ii. for the reversal.")
	assert contains_shakespeare_citation("See Act III, Scene 2 for the turn.")
	assert contains_shakespeare_citation("The line at ll. 42-45 is decisive.")
	assert not contains_shakespeare_citation("The critic offers no citation at all.")
	assert not contains_shakespeare_citation("Mr. Smith, e.g., disagrees entirely.")
	assert not contains_shakespeare_citation(
		"Section 1.E.1 of the Project Gutenberg License."
	)


def test_prompt_examples_only_use_schema_labels() -> None:
	parser = PydanticOutputParser(pydantic_object=LitAnalysisList)
	prompt = build_relation_system_prompt(
		["Macbeth"],
		parser.get_format_instructions(),
	)
	example_labels = re.findall(r'"relationship_type": \["([^"]+)"\]', prompt)

	assert example_labels
	assert set(example_labels).issubset(set(ALLOWED_RELATION_LABELS))
	assert set(ALLOWED_RELATION_LABELS) == {label.value for label in RelationLabel}
	assert "$work_titles" not in prompt
	assert "$allowed_relation_labels" not in prompt
	assert "$format_instructions" not in prompt


def test_extract_relations_from_text_retries_on_parse_failure() -> None:
	extractor = FakeExtractor(["not json", _valid_output()])

	relations = extract_relations_from_text(
		"The critic says blood imagery at 2.1.33 foreshadows the banquet vision at 3.4.50.",
		["Macbeth"],
		extractor=extractor,
		max_retries=1,
	)

	assert len(relations) == 1
	assert [
		getattr(label, "value", label) for label in relations[0].relationship_type
	] == ["foreshadow"]
	assert len(extractor.calls) == 2
	assert any(
		"Your previous output was invalid." in str(message.content)
		for message in extractor.calls[-1]
	)


def test_extract_relations_cli_writes_jsonl(
	tmp_path: Path,
	monkeypatch,
) -> None:
	input_file = tmp_path / "review.pdf"
	input_file.write_text("placeholder", encoding="utf-8")

	init_calls: list[str] = []
	timeout_calls: list[tuple[float, int]] = []
	extract_calls: list[str] = []

	class DummyExtractor:
		def __init__(
			self,
			model_id: str,
			*,
			timeout_seconds: float,
			timeout_retries: int,
		) -> None:
			init_calls.append(model_id)
			timeout_calls.append((timeout_seconds, timeout_retries))

		def close(self) -> None:
			return None

	def fake_extract_text(
		_: Path,
		*,
		pdf_ocr_mode: str,
		pdf_ocr_backend: str,
		pdf_ocr_language: str,
		pdf_ocr_workers: int,
	) -> str:
		assert pdf_ocr_mode == "full_ocr"
		assert pdf_ocr_backend == "tesseract"
		assert pdf_ocr_language == "eng"
		assert pdf_ocr_workers == 8
		return "raw review text"

	def fake_select_relevant_passages(_: str, target_works: list[str]) -> list[str]:
		assert target_works == ["Macbeth"]
		return ["passage one", "passage two"]

	def fake_extract_relations_from_text(
		text: str,
		target_works: list[str],
		extractor: object,
		max_retries: int = 3,
	) -> list[LitAnalysis]:
		assert isinstance(extractor, DummyExtractor)
		assert target_works == ["Macbeth"]
		assert max_retries == 3
		extract_calls.append(text)
		return [
			LitAnalysis(
				source_element="Blood imagery",
				target_element="Banquet vision",
				relationship_type=["foreshadow"],
				source_citation=(2, 1, 33),
				target_citation=(3, 4, 50),
				justification_quote=f"quote for {text}",
				confidence=0.9,
			)
		]

	monkeypatch.setattr(secondary_cli, "TRAPIRelationExtractor", DummyExtractor)
	monkeypatch.setattr(secondary_cli, "extract_text", fake_extract_text)
	monkeypatch.setattr(
		secondary_cli,
		"select_relevant_passages",
		fake_select_relevant_passages,
	)
	monkeypatch.setattr(
		secondary_cli,
		"extract_relations_from_text",
		fake_extract_relations_from_text,
	)

	runner = CliRunner()
	result = runner.invoke(
		secondary_cli.secondary_app,
		[
			"extract_relations",
			str(input_file),
			"--target-works",
			"Macbeth",
			"--model",
			"test-model",
			"--timeout-seconds",
			"240",
			"--timeout-retries",
			"2",
			"--pdf-ocr-mode",
			"full_ocr",
			"--pdf-ocr-backend",
			"tesseract",
			"--pdf-ocr-language",
			"eng",
			"--pdf-ocr-workers",
			"8",
			"-o",
			str(tmp_path),
		],
	)

	assert result.exit_code == 0, result.stdout
	assert init_calls == ["test-model"]
	assert timeout_calls == [(240.0, 2)]
	assert extract_calls == ["passage one", "passage two"]

	output_file = tmp_path / "review - Macbeth.jsonl"
	lines = output_file.read_text(encoding="utf-8").splitlines()

	assert len(lines) == 2
	assert all(json.loads(line)["source_file"] == "review" for line in lines)
