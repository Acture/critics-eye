"""Parity between the upstream MotifBench scorer and the Harbor verifier."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from scripts.motifbench_harbor.evaluate import aggregate_rewards, score_paths
from scripts.motifbench_harbor.run_parity_mini_swe_agent import (
	env_data_dir,
	format_mean_sem,
	sample_sem,
)
from scripts.motifbench_harbor.tasks import MotifBenchAdapter

REPO_ROOT = Path(__file__).resolve().parents[1]
HARBOR_PKG = REPO_ROOT / "scripts" / "motifbench_harbor"
VERIFIER_PATH = HARBOR_PKG / "task-template" / "tests" / "verifier.py"
METRIC_PATH = HARBOR_PKG / "metric.py"

REWARD_FLOAT_KEYS = (
	"reward",
	"score_hit_at_1",
	"score_hit_at_5",
	"score_hit_at_10",
	"score_mrr_at_20",
	"score_ndcg_at_10",
	"score_sum",
)


def _require(path: Path) -> Path:
	if not path.exists():
		pytest.fail(f"Required MotifBench Harbor path is missing: {path}")
	return path


def _load_module(path: Path, name: str) -> ModuleType:
	spec = importlib.util.spec_from_file_location(name, path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Cannot load {path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def _reward_close(left: dict[str, object], right: dict[str, object]) -> None:
	assert int(cast(int, left["total_queries"])) == int(
		cast(int, right["total_queries"])
	)
	for key in REWARD_FLOAT_KEYS:
		assert math.isclose(
			float(cast(float, left[key])),
			float(cast(float, right[key])),
			rel_tol=0.0,
			abs_tol=1e-9,
		), key


def _run_harbor_verifier(
	ground_truth: Path,
	predictions: Path,
	output_dir: Path,
) -> dict[str, object]:
	module = cast(
		Any,
		_load_module(_require(VERIFIER_PATH), "motifbench_harbor_verifier_under_test"),
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	module.GROUND_TRUTH_PATH = ground_truth
	module.PREDICTIONS_PATH = predictions
	module.REWARD_PATH = output_dir / "reward.json"
	module.DETAILS_PATH = output_dir / "grading_details.json"
	main = module.main
	if not callable(main):
		raise TypeError("Harbor verifier main is not callable")
	main()
	reward = json.loads((output_dir / "reward.json").read_text(encoding="utf-8"))
	details = json.loads(
		(output_dir / "grading_details.json").read_text(encoding="utf-8")
	)
	if not isinstance(reward, dict) or not isinstance(details, dict):
		raise TypeError("Harbor verifier did not write JSON objects")
	return {"reward": reward, "details": details}


def _run_harbor_metric(jsonl: Path, output: Path) -> dict[str, object]:
	loaded = cast(
		Any, _load_module(_require(METRIC_PATH), "motifbench_harbor_metric_under_test")
	)
	main = loaded.main
	if not callable(main):
		raise TypeError("Harbor metric main is not callable")
	main(jsonl, output)
	raw = json.loads(output.read_text(encoding="utf-8"))
	if not isinstance(raw, dict):
		raise TypeError("Harbor metric did not write a JSON object")
	return {str(key): value for key, value in raw.items()}


def _write_ground_truth(
	path: Path, valid_ids: list[str], queries: list[dict[str, object]]
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps({"valid_motif_ids": valid_ids, "queries": queries}, indent=2) + "\n",
		encoding="utf-8",
	)


def _write_predictions(path: Path, payload: object) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
	return path


def _oracle_predictions(task_dir: Path) -> dict[str, list[str]]:
	ground_truth = json.loads(
		(task_dir / "tests" / "ground_truth.json").read_text(encoding="utf-8")
	)
	if not isinstance(ground_truth, dict):
		raise TypeError(task_dir)
	queries = ground_truth["queries"]
	if not isinstance(queries, list):
		raise TypeError(task_dir)
	predictions: dict[str, list[str]] = {}
	for query in queries:
		if not isinstance(query, dict):
			raise TypeError(task_dir)
		gold = query["gold_motif_ids"]
		if not isinstance(gold, list):
			raise TypeError(task_dir)
		predictions[str(query["line_id"])] = [str(mid) for mid in gold]
	return predictions


def _per_query(details: object) -> dict[str, object]:
	if not isinstance(details, dict):
		raise TypeError("details must be an object")
	per_query = details["per_query"]
	if not isinstance(per_query, list) or not isinstance(per_query[0], dict):
		raise TypeError("per_query must be a list of objects")
	return {str(key): value for key, value in per_query[0].items()}


def test_sample_sem_matches_harbor_definition() -> None:
	assert sample_sem([1.0]) is None
	assert format_mean_sem([1.0]) == "1.0000 ± undefined (n=1)"
	assert sample_sem([1.0, 3.0]) == pytest.approx(1.0)
	assert format_mean_sem([1.0, 3.0]) == "2.0000 ± 1.0000"


def test_oracle_predictions_match_harbor_verifier(tmp_path: Path) -> None:
	"""All 12 generated oracle tasks score 1.0000 and agree with verifier.py and metric.py."""
	tasks_root = tmp_path / "generated_tasks"
	MotifBenchAdapter(
		tasks_root,
		data_dir=env_data_dir(),
		granularity="scene-group",
		split="parity",
		overwrite=True,
	).run()
	task_dirs = sorted(
		path
		for path in tasks_root.iterdir()
		if (path / "tests" / "ground_truth.json").is_file()
	)
	assert len(task_dirs) == 12
	upstream_rewards: list[dict[str, float | int]] = []
	harbor_rewards: list[dict[str, object]] = []
	total_queries = 0
	for task_dir in task_dirs:
		predictions = _oracle_predictions(task_dir)
		pred_path = _write_predictions(
			tmp_path / task_dir.name / "predictions.json",
			predictions,
		)
		ground_truth = task_dir / "tests" / "ground_truth.json"
		upstream = score_paths(ground_truth, pred_path)
		harbor = _run_harbor_verifier(
			ground_truth, pred_path, tmp_path / task_dir.name / "harbor"
		)
		harbor_reward = harbor["reward"]
		if not isinstance(harbor_reward, dict):
			raise TypeError("reward must be an object")
		_reward_close(cast(dict[str, object], upstream.reward), harbor_reward)
		assert float(upstream.reward["reward"]) == pytest.approx(1.0)
		assert int(upstream.reward["total_queries"]) == len(predictions)
		upstream_rewards.append(upstream.reward)
		harbor_rewards.append(harbor_reward)
		total_queries += len(predictions)
	assert total_queries == 270
	upstream_micro = aggregate_rewards(upstream_rewards)
	assert upstream_micro["total_queries"] == 270
	assert float(upstream_micro["micro_hit_at_5"]) == pytest.approx(1.0)
	jsonl = tmp_path / "rewards.jsonl"
	jsonl.write_text(
		"\n".join(json.dumps(reward) for reward in harbor_rewards) + "\n",
		encoding="utf-8",
	)
	harbor_micro = _run_harbor_metric(jsonl, tmp_path / "metric.json")
	assert int(cast(int, harbor_micro["total_queries"])) == 270
	assert math.isclose(
		float(cast(float, harbor_micro["micro_hit_at_5"])),
		float(upstream_micro["micro_hit_at_5"]),
		rel_tol=0.0,
		abs_tol=1e-9,
	)


def test_edge_cases_match_harbor_verifier(tmp_path: Path) -> None:
	valid_ids = [f"motif_{index}" for index in range(25)]
	cases: list[dict[str, object]] = [
		{
			"name": "duplicates",
			"gold": ["motif_2"],
			"predictions": ["motif_1", "bogus", "motif_1", "motif_2"],
			"best_rank": 2,
			"top5": ["motif_1", "motif_2"],
		},
		{
			"name": "invalid",
			"gold": ["motif_2"],
			"predictions": ["not-a-motif", "motif_2"],
			"best_rank": 1,
			"top5": ["motif_2"],
		},
		{
			"name": "truncated",
			"gold": ["motif_20"],
			"predictions": [f"motif_{index}" for index in range(25)],
			"best_rank": None,
			"top5": [f"motif_{index}" for index in range(5)],
		},
		{
			"name": "duplicate_does_not_consume_cap",
			"gold": ["motif_19"],
			"predictions": [
				"motif_0",
				"motif_0",
				*[f"motif_{index}" for index in range(1, 21)],
			],
			"best_rank": 20,
			"top5": [f"motif_{index}" for index in range(5)],
		},
		{
			"name": "partial_rank_2",
			"gold": ["motif_2"],
			"predictions": ["motif_0", "motif_2", "motif_3"],
			"best_rank": 2,
			"top5": ["motif_0", "motif_2", "motif_3"],
		},
	]
	for case in cases:
		name = str(case["name"])
		gold = case["gold"]
		predicted = case["predictions"]
		if not isinstance(gold, list) or not isinstance(predicted, list):
			raise TypeError(name)
		ground_truth = tmp_path / name / "tests" / "ground_truth.json"
		_write_ground_truth(
			ground_truth,
			valid_ids,
			[{"line_id": "ftln-0001", "citation": "1.1.1", "gold_motif_ids": gold}],
		)
		pred_path = _write_predictions(
			tmp_path / name / "predictions.json",
			{"ftln-0001": predicted},
		)
		upstream = score_paths(ground_truth, pred_path)
		harbor = _run_harbor_verifier(
			ground_truth, pred_path, tmp_path / name / "harbor"
		)
		harbor_reward = harbor["reward"]
		if not isinstance(harbor_reward, dict):
			raise TypeError(name)
		_reward_close(cast(dict[str, object], upstream.reward), harbor_reward)
		upstream_query = _per_query(upstream.details)
		harbor_query = _per_query(harbor["details"])
		assert upstream_query["best_rank"] == case["best_rank"]
		assert harbor_query["best_rank"] == case["best_rank"]
		assert upstream_query["top5_predicted"] == case["top5"]
		assert harbor_query["top5_predicted"] == case["top5"]

	partial_gt = tmp_path / "partial" / "tests" / "ground_truth.json"
	_write_ground_truth(
		partial_gt,
		valid_ids,
		[{"line_id": "ftln-0001", "citation": "1.1.1", "gold_motif_ids": ["motif_2"]}],
	)
	partial = score_paths(
		partial_gt,
		_write_predictions(
			tmp_path / "partial" / "predictions.json",
			{"ftln-0001": ["motif_0", "motif_2"]},
		),
	)
	assert float(partial.reward["score_hit_at_1"]) == pytest.approx(0.0)
	assert float(partial.reward["score_hit_at_5"]) == pytest.approx(1.0)
	assert float(partial.reward["score_mrr_at_20"]) == pytest.approx(0.5)
	assert float(partial.reward["score_ndcg_at_10"]) == pytest.approx(
		1.0 / math.log2(3), abs=1e-6
	)

	missing_gt = tmp_path / "missing" / "tests" / "ground_truth.json"
	_write_ground_truth(
		missing_gt,
		valid_ids,
		[{"line_id": "ftln-0001", "citation": "1.1.1", "gold_motif_ids": ["motif_2"]}],
	)
	missing = score_paths(missing_gt, tmp_path / "missing" / "absent.json")
	missing_harbor = _run_harbor_verifier(
		missing_gt,
		tmp_path / "missing" / "absent.json",
		tmp_path / "missing" / "harbor",
	)
	missing_reward = missing_harbor["reward"]
	if not isinstance(missing_reward, dict):
		raise TypeError("missing reward")
	_reward_close(cast(dict[str, object], missing.reward), missing_reward)
	assert float(missing.reward["reward"]) == pytest.approx(0.0)
	assert missing.details["load_error"] is not None

	malformed_path = tmp_path / "malformed.json"
	malformed_path.write_text("{not json", encoding="utf-8")
	malformed = score_paths(missing_gt, malformed_path)
	malformed_harbor = _run_harbor_verifier(
		missing_gt, malformed_path, tmp_path / "malformed-harbor"
	)
	malformed_reward = malformed_harbor["reward"]
	if not isinstance(malformed_reward, dict):
		raise TypeError("malformed reward")
	_reward_close(cast(dict[str, object], malformed.reward), malformed_reward)
	assert float(malformed.reward["reward"]) == pytest.approx(0.0)
	assert malformed.details["load_error"] is not None


def test_metric_null_trial_fails_closed(tmp_path: Path) -> None:
	jsonl = tmp_path / "rewards.jsonl"
	jsonl.write_text('{"score_sum": 1.0, "total_queries": 2}\nnull\n', encoding="utf-8")
	harbor = _run_harbor_metric(jsonl, tmp_path / "metric.json")
	upstream = aggregate_rewards(
		[{"score_sum": 1.0, "total_queries": 2, "reward": 0.5}, None]
	)
	assert harbor == {"num_failed_trials": 1, "total_queries_scored": 2}
	assert upstream["num_failed_trials"] == 1
	assert upstream["total_queries_scored"] == 2


def test_run_parity_oracle_scores_one(tmp_path: Path) -> None:
	"""``--mode oracle`` scores 1.0000 on all 12 scene-group tasks and 270 queries."""
	script = HARBOR_PKG / "run_parity_mini_swe_agent.py"
	cmd = [
		sys.executable,
		str(script),
		"--mode",
		"oracle",
		"--output-dir",
		str(tmp_path),
		"--granularity",
		"scene-group",
		"--split",
		"parity",
		"--runs",
		"1",
	]
	data_dir = env_data_dir()
	if data_dir is not None:
		cmd.extend(["--data-dir", str(data_dir)])
	completed = subprocess.run(
		cmd,
		check=False,
		capture_output=True,
		text=True,
	)
	assert completed.returncode == 0, completed.stderr + completed.stdout
	summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
	assert summary["agent_invoked"] is False
	assert summary["n_tasks"] == 12
	assert summary["total_queries"] == 270
	assert summary["per_run_micro_hit_at_5"] == [1.0]
	assert float(summary["mean"]) == pytest.approx(1.0)
	trial = json.loads(
		(tmp_path / "result_original_trial1.json").read_text(encoding="utf-8")
	)
	assert trial["n_errors"] == 0
	assert float(trial["metrics"]["micro_hit_at_5"]) == pytest.approx(1.0)
	assert int(trial["metrics"]["total_queries"]) == 270
	assert len(trial["tasks"]) == 12
	for task in trial["tasks"]:
		assert float(task["reward"]["reward"]) == pytest.approx(1.0)
	assert "1.0000" in completed.stdout


def test_dry_run_stages_instruction_without_subprocess(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import scripts.motifbench_harbor.run_parity_mini_swe_agent as runner

	def forbid_subprocess(*_args: object, **_kwargs: object) -> None:
		raise AssertionError("dry-run invoked a subprocess")

	monkeypatch.setattr(runner.subprocess, "run", forbid_subprocess)
	argv = [
		"--mode",
		"dry-run",
		"--output-dir",
		str(tmp_path),
		"--granularity",
		"scene-group",
		"--split",
		"parity",
		"--limit",
		"1",
	]
	data_dir = env_data_dir()
	if data_dir is not None:
		argv.extend(["--data-dir", str(data_dir)])
	manifest = runner.run(argv)
	assert manifest["agent_invoked"] is False
	assert manifest["n_tasks"] == 1
	tasks = manifest["tasks"]
	if not isinstance(tasks, list) or not isinstance(tasks[0], dict):
		raise TypeError("dry-run tasks must be a list of objects")
	instruction = Path(str(tasks[0]["instruction_path"]))
	text = instruction.read_text(encoding="utf-8")
	assert text.startswith("# MotifBench:")
	assert "/app/macbeth.txt" in text
	app_dir = Path(str(tasks[0]["app_dir"]))
	for name in ("macbeth.txt", "macbeth.pdf", "motifs.json", "queries.json"):
		assert (app_dir / name).is_file()
	assert not (tmp_path / "result_original_trial1.json").exists()
