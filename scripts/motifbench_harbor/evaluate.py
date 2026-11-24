# /// script
# dependencies = []
# ///
"""Deterministic MotifBench scorer aligned with the Harbor verifier.

The per-query formulas match ``adapters/motifbench/src/motifbench/task-template/tests/verifier.py``.
Dataset aggregation matches ``adapters/motifbench/src/motifbench/metric.py``:
micro-averaged Hit@5 is the sum of per-task ``score_sum`` divided by the sum of
``total_queries``, and a null trial reward fails the aggregate closed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MAX_RANK_EVAL = 20


def fail_safe_reward(total_queries: int) -> dict[str, float | int]:
	"""Zero reward written before grading. ``reward`` is the first key."""
	return {
		"reward": 0.0,
		"score_hit_at_1": 0.0,
		"score_hit_at_5": 0.0,
		"score_hit_at_10": 0.0,
		"score_mrr_at_20": 0.0,
		"score_ndcg_at_10": 0.0,
		"score_sum": 0.0,
		"total_queries": total_queries,
	}


def parse_predictions(
	raw: object, source: str
) -> tuple[dict[str, list[str]], str | None]:
	"""Parse a predictions object the way the Harbor verifier parses JSON."""
	if not isinstance(raw, dict):
		return {}, f"Expected a JSON object in {source}, got {type(raw).__name__}"
	parsed: dict[str, list[str]] = {}
	for key, val in raw.items():
		if isinstance(val, list):
			parsed[str(key)] = [
				str(item).strip() for item in val if isinstance(item, str)
			]
		elif isinstance(val, str):
			parsed[str(key)] = [val.strip()]
	return parsed, None


def load_predictions(path: Path) -> tuple[dict[str, list[str]], str | None]:
	"""Load ``predictions.json``. Missing and malformed files score as empty."""
	if not path.exists():
		return {}, f"Missing predictions file at {path}"
	try:
		raw = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as exc:
		return {}, f"Invalid JSON in {path}: {exc}"
	return parse_predictions(raw, str(path))


def clean_ranking(raw_ids: list[str], valid_motif_ids: set[str]) -> list[str]:
	"""Drop unknown ids, keep the first copy of each id, and stop at 20."""
	seen: set[str] = set()
	cleaned: list[str] = []
	for mid in raw_ids:
		if mid in valid_motif_ids and mid not in seen:
			seen.add(mid)
			cleaned.append(mid)
			if len(cleaned) >= MAX_RANK_EVAL:
				break
	return cleaned


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float:
	"""Binary nDCG@k. Empty gold scores 0, matching the Harbor verifier."""
	if not gold:
		return 0.0
	dcg = 0.0
	for idx, mid in enumerate(ranked[:k], start=1):
		if mid in gold:
			dcg += 1.0 / math.log2(idx + 1)
	ideal_hits = min(len(gold), k)
	idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
	return dcg / idcg if idcg > 0.0 else 0.0


def _as_query_list(ground_truth: dict[str, object]) -> list[dict[str, object]]:
	raw_queries = ground_truth.get("queries")
	if not isinstance(raw_queries, list):
		raise TypeError("ground truth 'queries' must be a list")
	queries: list[dict[str, object]] = []
	for item in raw_queries:
		if not isinstance(item, dict):
			raise TypeError("each ground-truth query must be an object")
		queries.append({str(key): value for key, value in item.items()})
	return queries


def _valid_motif_ids(ground_truth: dict[str, object]) -> set[str]:
	raw_ids = ground_truth.get("valid_motif_ids")
	if not isinstance(raw_ids, list):
		raise TypeError("ground truth 'valid_motif_ids' must be a list")
	return {str(item) for item in raw_ids}


@dataclass(frozen=True)
class TaskScore:
	"""Verifier reward payload plus the grading-details object."""

	reward: dict[str, float | int]
	details: dict[str, object]


def score_ground_truth(
	ground_truth: dict[str, object],
	predictions: dict[str, list[str]],
	load_error: str | None = None,
) -> TaskScore:
	"""Score one task. Rounding and key order follow ``verifier.py``."""
	valid_motif_ids = _valid_motif_ids(ground_truth)
	queries = _as_query_list(ground_truth)
	total_queries = len(queries)

	hit1_sum = 0.0
	hit5_sum = 0.0
	hit10_sum = 0.0
	mrr20_sum = 0.0
	ndcg10_sum = 0.0
	num_answered = 0
	per_query: list[dict[str, object]] = []

	for query in queries:
		line_id = str(query["line_id"])
		raw_gold = query.get("gold_motif_ids")
		gold_list = [str(mid) for mid in raw_gold] if isinstance(raw_gold, list) else []
		gold_set = set(gold_list)
		ranked = clean_ranking(predictions.get(line_id, []), valid_motif_ids)
		if ranked:
			num_answered += 1

		best_rank: int | None = None
		for idx, mid in enumerate(ranked, start=1):
			if mid in gold_set:
				best_rank = idx
				break

		hit1 = 1.0 if best_rank is not None and best_rank <= 1 else 0.0
		hit5 = 1.0 if best_rank is not None and best_rank <= 5 else 0.0
		hit10 = 1.0 if best_rank is not None and best_rank <= 10 else 0.0
		mrr = (1.0 / best_rank) if best_rank is not None and best_rank <= 20 else 0.0
		ndcg = ndcg_at_k(ranked, gold_set, k=10)

		hit1_sum += hit1
		hit5_sum += hit5
		hit10_sum += hit10
		mrr20_sum += mrr
		ndcg10_sum += ndcg

		per_query.append(
			{
				"line_id": line_id,
				"citation": query.get("citation"),
				"gold_count": len(gold_set),
				"best_rank": best_rank,
				"hit_at_1": hit1,
				"hit_at_5": hit5,
				"hit_at_10": hit10,
				"mrr_at_20": round(mrr, 6),
				"ndcg_at_10": round(ndcg, 6),
				"num_valid_preds": len(ranked),
				"top5_predicted": ranked[:5],
			}
		)

	denom = float(total_queries) if total_queries else 1.0
	hit1 = hit1_sum / denom
	hit5 = hit5_sum / denom
	hit10 = hit10_sum / denom
	mrr20 = mrr20_sum / denom
	ndcg10 = ndcg10_sum / denom
	reward: dict[str, float | int] = {
		"reward": round(hit5, 6),
		"score_hit_at_1": round(hit1, 6),
		"score_hit_at_5": round(hit5, 6),
		"score_hit_at_10": round(hit10, 6),
		"score_mrr_at_20": round(mrr20, 6),
		"score_ndcg_at_10": round(ndcg10, 6),
		"score_sum": round(hit5_sum, 6),
		"total_queries": total_queries,
	}
	details: dict[str, object] = {
		"load_error": load_error,
		"num_queries": total_queries,
		"num_answered": num_answered,
		"metrics": reward,
		"per_query": per_query,
	}
	return TaskScore(reward=reward, details=details)


def load_ground_truth(path: Path) -> dict[str, object]:
	"""Read a Harbor ``ground_truth.json`` object."""
	raw = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(raw, dict):
		raise TypeError(f"Expected a JSON object in {path}")
	return {str(key): value for key, value in raw.items()}


def score_paths(ground_truth_path: Path, predictions_path: Path) -> TaskScore:
	"""Score the two files the Harbor verifier reads."""
	ground_truth = load_ground_truth(ground_truth_path)
	predictions, load_error = load_predictions(predictions_path)
	return score_ground_truth(ground_truth, predictions, load_error)


def write_task_score(score: TaskScore, output_dir: Path) -> None:
	"""Write ``reward.json`` and ``grading_details.json`` with verifier formatting."""
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "reward.json").write_text(
		json.dumps(score.reward, indent=2) + "\n",
		encoding="utf-8",
	)
	(output_dir / "grading_details.json").write_text(
		json.dumps(score.details, indent=2) + "\n",
		encoding="utf-8",
	)


SECONDARY_REWARD_KEYS = (
	"score_hit_at_1",
	"score_hit_at_5",
	"score_hit_at_10",
	"score_mrr_at_20",
	"score_ndcg_at_10",
)


def weighted_task_means(
	rewards: Iterable[dict[str, float | int]],
) -> dict[str, float]:
	"""Query-weighted means of the per-task reward averages.

	Each task reward stores a mean, so the dataset mean is
	``sum(mean_i * n_i) / sum(n_i)``. Hit@5's exact micro average remains
	``score_sum / total_queries`` in ``aggregate_rewards``.
	"""
	totals = {key: 0.0 for key in SECONDARY_REWARD_KEYS}
	total_queries = 0
	for reward in rewards:
		count = int(reward["total_queries"])
		total_queries += count
		for key in SECONDARY_REWARD_KEYS:
			totals[key] += float(reward[key]) * count
	if total_queries == 0:
		return {key: 0.0 for key in SECONDARY_REWARD_KEYS}
	return {key: value / total_queries for key, value in totals.items()}


def aggregate_rewards(
	rewards: Iterable[dict[str, float | int] | None],
) -> dict[str, float | int]:
	"""Micro-average Hit@5 the way ``metric.py`` reduces a reward JSONL."""
	score_sum = 0.0
	total_queries = 0
	num_failed = 0
	for reward in rewards:
		if reward is None:
			num_failed += 1
			continue
		score_sum += float(reward["score_sum"])
		total_queries += int(reward["total_queries"])
	if num_failed:
		return {
			"num_failed_trials": num_failed,
			"total_queries_scored": total_queries,
		}
	micro_hit5 = score_sum / total_queries if total_queries else 0.0
	return {
		"micro_hit_at_5": micro_hit5,
		"total_queries": total_queries,
	}


def _task_dirs(tasks_root: Path) -> list[Path]:
	dirs = [
		child
		for child in sorted(tasks_root.iterdir())
		if child.is_dir() and (child / "tests" / "ground_truth.json").is_file()
	]
	if not dirs:
		raise FileNotFoundError(
			f"No task directories with ground_truth.json under {tasks_root}"
		)
	return dirs


def _find_predictions(predictions_root: Path, task_id: str) -> Path:
	candidates = (
		predictions_root / task_id / "workspace" / "predictions.json",
		predictions_root / task_id / "predictions.json",
		predictions_root / f"{task_id}.json",
	)
	for candidate in candidates:
		if candidate.exists():
			return candidate
	return candidates[0]


def evaluate_tasks(
	tasks_root: Path,
	predictions_root: Path,
	output_dir: Path,
) -> dict[str, object]:
	"""Score every task directory and write per-task rewards plus ``summary.json``."""
	output_dir.mkdir(parents=True, exist_ok=True)
	rewards_for_metric: list[dict[str, float | int] | None] = []
	task_rows: list[dict[str, object]] = []
	jsonl_lines: list[str] = []
	for task_dir in _task_dirs(tasks_root):
		predictions_path = _find_predictions(predictions_root, task_dir.name)
		score = score_paths(task_dir / "tests" / "ground_truth.json", predictions_path)
		task_out = output_dir / task_dir.name
		write_task_score(score, task_out)
		rewards_for_metric.append(score.reward)
		jsonl_lines.append(json.dumps(score.reward))
		task_rows.append(
			{
				"task_id": task_dir.name,
				"predictions": str(predictions_path),
				"reward": score.reward,
			}
		)
	(output_dir / "rewards.jsonl").write_text(
		"\n".join(jsonl_lines) + "\n",
		encoding="utf-8",
	)
	summary: dict[str, object] = {
		"metric": aggregate_rewards(rewards_for_metric),
		"n_tasks": len(task_rows),
		"tasks": task_rows,
	}
	(output_dir / "summary.json").write_text(
		json.dumps(summary, indent=2) + "\n",
		encoding="utf-8",
	)
	return summary


def _print_reward(score: TaskScore) -> None:
	reward = score.reward
	total = int(reward["total_queries"])
	answered_raw = score.details["num_answered"]
	if not isinstance(answered_raw, int):
		raise TypeError("num_answered must be an int")
	answered = answered_raw
	print(
		f"MotifBench reward (Hit@5) = {float(reward['reward']):.4f} | "
		f"Hit@1 = {float(reward['score_hit_at_1']):.4f} | "
		f"Hit@10 = {float(reward['score_hit_at_10']):.4f} | "
		f"MRR@20 = {float(reward['score_mrr_at_20']):.4f} | "
		f"nDCG@10 = {float(reward['score_ndcg_at_10']):.4f} "
		f"over {total} queries ({answered} answered)"
	)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(
		description="Score MotifBench predictions with the Harbor verifier metrics."
	)
	parser.add_argument(
		"--task-dir", type=Path, default=None, help="One Harbor task directory."
	)
	parser.add_argument(
		"--predictions",
		type=Path,
		default=None,
		help="predictions.json for --task-dir.",
	)
	parser.add_argument(
		"--tasks-root",
		type=Path,
		default=None,
		help="Directory of Harbor task directories.",
	)
	parser.add_argument(
		"--predictions-root",
		type=Path,
		default=None,
		help="Per-task predictions, as <id>/workspace/predictions.json or <id>/predictions.json.",
	)
	parser.add_argument("--output-dir", type=Path, required=True)
	args = parser.parse_args(argv)

	single = args.task_dir is not None or args.predictions is not None
	multi = args.tasks_root is not None or args.predictions_root is not None
	if single and multi:
		parser.error(
			"Pass either --task-dir/--predictions or --tasks-root/--predictions-root."
		)
	if single:
		if args.task_dir is None or args.predictions is None:
			parser.error("--task-dir and --predictions are both required.")
		score = score_paths(
			args.task_dir / "tests" / "ground_truth.json", args.predictions
		)
		write_task_score(score, args.output_dir)
		summary = {
			"metric": aggregate_rewards([score.reward]),
			"n_tasks": 1,
			"tasks": [{"task_id": args.task_dir.name, "reward": score.reward}],
		}
		(args.output_dir / "summary.json").write_text(
			json.dumps(summary, indent=2) + "\n",
			encoding="utf-8",
		)
		_print_reward(score)
		return
	if args.tasks_root is None or args.predictions_root is None:
		parser.error(
			"Provide --task-dir and --predictions, or --tasks-root and --predictions-root."
		)
	summary = evaluate_tasks(args.tasks_root, args.predictions_root, args.output_dir)
	metric = summary["metric"]
	n_tasks = summary["n_tasks"]
	micro = metric.get("micro_hit_at_5") if isinstance(metric, dict) else None
	total = metric.get("total_queries") if isinstance(metric, dict) else None
	if (
		isinstance(micro, (int, float))
		and not isinstance(micro, bool)
		and isinstance(total, int)
		and isinstance(n_tasks, int)
	):
		print(
			f"micro Hit@5 = {float(micro):.4f} over {total} queries ({n_tasks} tasks)"
		)
	else:
		print(json.dumps(metric))


if __name__ == "__main__":
	main()
