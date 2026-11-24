# /// script
# dependencies = []
# ///
"""Scenario-2 parity runner for MotifBench against Harbor's mini-swe-agent.

Fetches ``Acture/MotifBench`` at revision
``12fadc6a46aa8d2fcc7848777574122afeccf59f`` (or reads ``--data-dir``) and
generates the scene-group tasks itself. It does not import a Harbor checkout.
Each task's read-only inputs are ``/app/macbeth.txt``, ``/app/macbeth.pdf``,
``/app/motifs.json``, and ``/app/queries.json``, and ``instruction.md`` matches
the adapter template.

``--mode oracle`` writes the critic gold lists and scores them.
``--mode dry-run`` stages that layout and prints the ``mini-swe-agent`` command
without launching Docker. ``--mode mini-swe`` is the paid path and is never
the default. It runs ``mini-swe-agent==2.2.8`` with ``openai/gpt-5.4-mini``
inside a fresh container per task. ``OPENAI_API_KEY`` is taken from the
process environment and passed to Docker with ``-e OPENAI_API_KEY``; this
module never writes the key.

The upstream scorer is ``scripts/motifbench_harbor/evaluate.py``. Each run
writes ``result_original_trial{n}.json`` plus a mean ± sample SEM summary.
Sample SEM is ``sqrt(Σ (x_i - mean)^2 / (n * (n - 1)))`` for ``n >= 2``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from scripts.motifbench_harbor.tasks import (
	DEFAULT_HF_REPO,
	DEFAULT_HF_REVISION,
	Granularity,
	MotifBenchAdapter,
	SplitMode,
)

AGENT_VERSION = "2.2.8"
DEFAULT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_IMAGE = f"motifbench-mini-swe-agent:{AGENT_VERSION}"
DEFAULT_TIMEOUT_SEC = 1200
DEFAULT_CONCURRENCY = 4
TRAJECTORY_NAME = "mini-swe-agent.trajectory.json"
CONTAINER_TRAJECTORY = f"/logs/agent/{TRAJECTORY_NAME}"
CONFIG_PATH = Path(__file__).resolve().parent / "mini_swe_agent.yaml"
DOCKERFILE_PATH = Path(__file__).resolve().parent / "parity.Dockerfile"
_SECRET_RE = re.compile(rb"sk-[A-Za-z0-9_\-]{8,}")
_SECRET_TEXT_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}|OPENAI_API_KEY=\S+")


class _TaskBuilder(Protocol):
	def run(self) -> list[Path]: ...


def sample_sem(values: list[float]) -> float | None:
	"""Sample standard error of the mean. Undefined when fewer than two runs exist."""
	count = len(values)
	if count < 2:
		return None
	mean = sum(values) / count
	squared = sum((value - mean) ** 2 for value in values)
	return math.sqrt(squared / (count * (count - 1)))


def format_mean_sem(values: list[float]) -> str:
	"""Harbor parity string. A single run has no sample SEM."""
	mean = sum(values) / len(values) if values else 0.0
	sem = sample_sem(values)
	if sem is None:
		return f"{mean:.4f} ± undefined (n={len(values)})"
	return f"{mean:.4f} ± {sem:.4f}"


def format_mean_sem_ascii(values: list[float]) -> str:
	"""Adapter JSON form of the sample mean and SEM."""
	mean = sum(values) / len(values) if values else 0.0
	sem = sample_sem(values)
	if sem is None:
		return f"{mean:.4f}"
	return f"{mean:.4f} +/- {sem:.4f}"


def scrub_text(text: str) -> str:
	"""Remove API-key shaped substrings before anything is written or printed."""
	return _SECRET_TEXT_RE.sub("[REDACTED]", text)


def scrub_tree(root: Path) -> int:
	"""Rewrite secret-shaped byte sequences in text logs. Returns the match count.

	Play files under ``app/`` and binary inputs are left untouched. Files the
	host cannot rewrite are skipped so a root-owned trajectory cannot abort
	the run.
	"""
	if not root.exists():
		return 0
	replaced = 0
	paths = [root] if root.is_file() else list(root.rglob("*"))
	for path in paths:
		if not path.is_file():
			continue
		if "app" in path.parts or path.suffix.lower() in {".pdf", ".pt", ".png", ".pyc"}:
			continue
		try:
			raw = path.read_bytes()
		except OSError:
			continue
		updated, count = _SECRET_RE.subn(b"[REDACTED]", raw)
		if count:
			try:
				path.write_bytes(updated)
			except OSError:
				continue
			replaced += count
	return replaced


def _relax_tree(work: Path) -> None:
	"""Let the host user read files the agent container wrote as root."""
	if not work.exists():
		return
	subprocess.run(
		[
			"docker",
			"run",
			"--rm",
			"-v",
			f"{work.resolve()}:/work",
			"python:3.13-slim",
			"chmod",
			"-R",
			"a+rwX",
			"/work",
		],
		capture_output=True,
		check=False,
	)


def env_data_dir() -> Path | None:
	"""Return ``MOTIFBENCH_DATA_DIR`` when it holds the cleaned export."""
	override = os.environ.get("MOTIFBENCH_DATA_DIR")
	if override is None or override == "":
		return None
	candidate = Path(override)
	required = (
		"macbeth_full.txt",
		"macbeth_full.pdf",
		"macbeth_clean.jsonl",
		"motifs_clean.json",
	)
	missing = [name for name in required if not (candidate / name).is_file()]
	if missing:
		raise FileNotFoundError(
			f"MOTIFBENCH_DATA_DIR={candidate} is missing: {', '.join(missing)}"
		)
	return candidate


def _as_granularity(value: str) -> Granularity:
	if value == "scene-group" or value == "act" or value == "query":
		return value
	raise ValueError(f"Unsupported granularity: {value}")


def _as_split(value: str) -> SplitMode:
	if value == "full" or value == "parity":
		return value
	raise ValueError(f"Unsupported split: {value}")


def load_task_builder(
	*,
	output_dir: Path,
	data_dir: Path | None,
	granularity: str,
	split: str,
	limit: int | None,
	task_ids: list[str] | None,
) -> _TaskBuilder:
	"""Configure the vendored generator. It fetches MotifBench when ``data_dir`` is omitted."""
	return MotifBenchAdapter(
		output_dir,
		data_dir=data_dir,
		granularity=_as_granularity(granularity),
		split=_as_split(split),
		limit=limit,
		overwrite=True,
		task_ids=task_ids,
	)


def cast_any(value: object) -> Any:
	"""Return a dynamically loaded module without a static import."""
	return value


def _sha256_text(path: Path) -> str:
	digest = hashlib.sha256()
	digest.update(path.read_bytes())
	return digest.hexdigest()


def _make_tree_writable(path: Path) -> None:
	if not path.exists():
		return
	for child in path.rglob("*"):
		if child.is_file():
			mode = child.stat().st_mode
			child.chmod(mode | stat.S_IWRITE | stat.S_IREAD)


def stage_task(task_dir: Path, destination: Path) -> dict[str, object]:
	"""Copy ``/app`` inputs and the verbatim Harbor instruction into ``destination``."""
	if destination.exists():
		_make_tree_writable(destination)
		shutil.rmtree(destination)
	app_dir = destination / "app"
	workspace = destination / "workspace"
	app_dir.mkdir(parents=True)
	workspace.mkdir(parents=True)
	environment = task_dir / "environment"
	for name in ("macbeth.txt", "macbeth.pdf", "motifs.json", "queries.json"):
		target = app_dir / name
		shutil.copy2(environment / name, target)
		target.chmod(stat.S_IREAD)
	instruction = destination / "instruction.md"
	shutil.copy2(task_dir / "instruction.md", instruction)
	return {
		"task_id": task_dir.name,
		"app_dir": str(app_dir),
		"workspace_dir": str(workspace),
		"instruction_path": str(instruction),
		"instruction_sha256": _sha256_text(instruction),
		"harbor_instruction_sha256": _sha256_text(task_dir / "instruction.md"),
	}


def oracle_predictions(task_dir: Path) -> dict[str, list[str]]:
	"""Gold motif lists from ``tests/ground_truth.json``, in stored order."""
	ground_truth = json.loads(
		(task_dir / "tests" / "ground_truth.json").read_text(encoding="utf-8")
	)
	if not isinstance(ground_truth, dict):
		raise TypeError(
			f"Expected an object in {task_dir / 'tests' / 'ground_truth.json'}"
		)
	queries = ground_truth.get("queries")
	if not isinstance(queries, list):
		raise TypeError("ground truth queries must be a list")
	predictions: dict[str, list[str]] = {}
	for query in queries:
		if not isinstance(query, dict):
			raise TypeError("each ground-truth query must be an object")
		line_id = str(query["line_id"])
		raw_gold = query.get("gold_motif_ids")
		if not isinstance(raw_gold, list):
			raise TypeError(f"{line_id} gold_motif_ids must be a list")
		predictions[line_id] = [str(mid) for mid in raw_gold]
	return predictions


def mini_swe_shell(
	*,
	model: str,
	instruction: str,
	config_in_container: str,
) -> str:
	"""Shell line Harbor's MiniSweAgent runs, including its LiteLLM timeout flag.

	Harbor prepends ``-c mini``, sets ``model.model_kwargs.timeout=3600``,
	then the shared YAML, then the API key from
	``MINI_SWE_AGENT_MODEL_API_KEY_JSON`` (shell-expanded, never inlined),
	then the chat-completions provider and model class. The YAML path is
	the copy mounted into the container.
	"""
	command = (
		'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
		'else export PATH="$HOME/.local/bin:$PATH"; fi; '
		f"mini-swe-agent --yolo --model={model} "
		f"--task={instruction} "
		f"--output={CONTAINER_TRAJECTORY} "
		"--cost-limit 1.0 "
		"-c mini "
		"-c model.model_kwargs.timeout=3600 "
		f"-c {config_in_container} "
		'-c "model.model_kwargs.api_key=$MINI_SWE_AGENT_MODEL_API_KEY_JSON" '
		"-c model.model_kwargs.custom_llm_provider=openai "
		"-c model.model_class=litellm "
		"--exit-immediately"
	)
	return command


def displayed_mini_swe_command(model: str) -> str:
	"""Command printed by dry-run. The task text is not inlined."""
	return (
		f"mini-swe-agent --yolo --model={model} --task=<instruction> "
		f"--output={CONTAINER_TRAJECTORY} --cost-limit 1.0 -c mini "
		"-c model.model_kwargs.timeout=3600 -c mini_swe_agent.yaml "
		'-c "model.model_kwargs.api_key=$MINI_SWE_AGENT_MODEL_API_KEY_JSON" '
		"-c model.model_kwargs.custom_llm_provider=openai "
		"-c model.model_class=litellm --exit-immediately"
	)


def _generated_task_dirs(tasks_root: Path) -> list[Path]:
	dirs = [
		child
		for child in sorted(tasks_root.iterdir())
		if child.is_dir() and (child / "tests" / "ground_truth.json").is_file()
	]
	dirs.sort(key=lambda path: path.name)
	return dirs


def _instruction_text(path: Path) -> str:
	"""Instruction text with Windows newline translation removed."""
	return path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def _check_instruction_identity(task_dir: Path, harbor_root: Path | None) -> None:
	"""Fail when a regenerated instruction differs from an optional reference dataset."""
	if harbor_root is None:
		return
	reference = (
		harbor_root / "datasets" / "motifbench" / task_dir.name / "instruction.md"
	)
	if not reference.is_file():
		return
	generated = _instruction_text(task_dir / "instruction.md")
	existing = _instruction_text(reference)
	if generated != existing:
		raise RuntimeError(
			f"instruction.md for {task_dir.name} does not match {reference}."
		)


def _load_evaluate() -> Any:
	return cast_any(importlib.import_module("scripts.motifbench_harbor.evaluate"))


def _score_task(
	task_dir: Path,
	predictions_path: Path,
	output_dir: Path,
) -> dict[str, float | int]:
	evaluate = _load_evaluate()
	score = evaluate.score_paths(
		task_dir / "tests" / "ground_truth.json", predictions_path
	)
	evaluate.write_task_score(score, output_dir)
	reward = score.reward
	if not isinstance(reward, dict):
		raise TypeError("score reward must be an object")
	parsed: dict[str, float | int] = {}
	for key, value in reward.items():
		if isinstance(value, bool) or not isinstance(value, (int, float)):
			raise TypeError(f"Reward field {key} is not numeric: {value!r}")
		parsed[str(key)] = value
	return parsed


def _typed_reward(reward: object) -> dict[str, float | int] | None:
	if not isinstance(reward, dict):
		return None
	parsed: dict[str, float | int] = {}
	for key, value in reward.items():
		if isinstance(value, bool) or not isinstance(value, (int, float)):
			raise TypeError(f"Reward field {key} is not numeric: {value!r}")
		parsed[str(key)] = value
	return parsed


def _macro_hit5(task_rows: list[dict[str, object]]) -> float | None:
	values: list[float] = []
	for row in task_rows:
		reward = _typed_reward(row.get("reward"))
		if reward is None:
			return None
		values.append(float(reward["reward"]))
	if not values:
		return None
	return sum(values) / len(values)


def _trial_payload(
	*,
	trial_index: int,
	mode: str,
	model: str | None,
	agent_version: str | None,
	granularity: str,
	split: str,
	task_rows: list[dict[str, object]],
	agent_invoked: bool,
	cost_usd: float,
) -> dict[str, object]:
	evaluate = _load_evaluate()
	typed_rewards: list[dict[str, float | int] | None] = [
		_typed_reward(row.get("reward")) for row in task_rows
	]
	metric = evaluate.aggregate_rewards(typed_rewards)
	if isinstance(metric, dict) and "micro_hit_at_5" in metric:
		complete = [reward for reward in typed_rewards if reward is not None]
		for key, value in evaluate.weighted_task_means(complete).items():
			metric.setdefault(key, value)
		macro = _macro_hit5(task_rows)
		if macro is not None:
			metric["macro_hit_at_5"] = macro
	return {
		"schema_version": 1,
		"source": "motifbench_harbor_upstream",
		"trial_index": trial_index,
		"mode": mode,
		"agent": f"mini-swe-agent@{agent_version}" if agent_version else mode,
		"model": model,
		"agent_version": agent_version,
		"granularity": granularity,
		"split": split,
		"n_tasks": len(task_rows),
		"n_errors": sum(1 for row in task_rows if row.get("error")),
		"agent_invoked": agent_invoked,
		"cost_usd": cost_usd,
		"metrics": metric,
		"tasks": task_rows,
	}


def ensure_image(image: str, rebuild: bool) -> None:
	"""Build the reusable parity image when it is missing."""
	if not rebuild:
		probe = subprocess.run(
			["docker", "image", "inspect", image],
			capture_output=True,
			check=False,
		)
		if probe.returncode == 0:
			return
	if not DOCKERFILE_PATH.is_file():
		raise FileNotFoundError(DOCKERFILE_PATH)
	completed = subprocess.run(
		[
			"docker",
			"build",
			"-f",
			str(DOCKERFILE_PATH),
			"-t",
			image,
			str(DOCKERFILE_PATH.parent),
		],
		capture_output=True,
		text=True,
		check=False,
	)
	if completed.returncode != 0:
		tail = scrub_text((completed.stderr or completed.stdout or "")[-1500:])
		raise RuntimeError(f"docker build rc={completed.returncode}: {tail}")


def _message_text(message: dict[str, object]) -> str:
	content = message.get("content")
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts: list[str] = []
		for item in content:
			if isinstance(item, str):
				parts.append(item)
			elif isinstance(item, dict):
				text = item.get("text")
				if isinstance(text, str):
					parts.append(text)
		return "\n".join(parts)
	return ""


def prompt_prefix(trajectory_path: Path) -> list[dict[str, str]]:
	"""Return the first system message and the first user message, in that order."""
	raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
	if not isinstance(raw, dict):
		raise TypeError(f"Expected an object in {trajectory_path}")
	messages = raw.get("messages")
	if not isinstance(messages, list):
		raise TypeError(f"{trajectory_path} has no messages list")
	found: list[dict[str, str]] = []
	for message in messages:
		if not isinstance(message, dict):
			continue
		role = message.get("role")
		if role == "system" and not any(item["role"] == "system" for item in found):
			found.append({"role": "system", "content": _message_text(message)})
		elif role == "user" and not any(item["role"] == "user" for item in found):
			found.append({"role": "user", "content": _message_text(message)})
		if len(found) == 2 and found[0]["role"] == "system":
			break
	return found


def compare_prompt_prefixes(left: Path, right: Path) -> dict[str, object]:
	"""Compare the first system and user messages of two trajectories."""
	left_messages = prompt_prefix(left)
	right_messages = prompt_prefix(right)
	rows: list[dict[str, object]] = []
	matched = len(left_messages) == len(right_messages) == 2
	for index, role in enumerate(("system", "user")):
		left_text = left_messages[index]["content"] if index < len(left_messages) else ""
		right_text = (
			right_messages[index]["content"] if index < len(right_messages) else ""
		)
		same = left_text == right_text and left_text != ""
		matched = matched and same
		rows.append(
			{
				"role": role,
				"equal": same,
				"left_sha256": hashlib.sha256(left_text.encode("utf-8")).hexdigest(),
				"right_sha256": hashlib.sha256(right_text.encode("utf-8")).hexdigest(),
				"left_chars": len(left_text),
				"right_chars": len(right_text),
			}
		)
	return {"equal": matched, "messages": rows}


def _load_cost(trajectory_path: Path) -> float:
	if not trajectory_path.is_file():
		return 0.0
	try:
		raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return 0.0
	if not isinstance(raw, dict):
		return 0.0
	info = raw.get("info")
	if not isinstance(info, dict):
		return 0.0
	stats = info.get("model_stats")
	if not isinstance(stats, dict):
		return 0.0
	cost = stats.get("instance_cost")
	if isinstance(cost, bool) or not isinstance(cost, (int, float)):
		return 0.0
	return float(cost)


def _run_container(
	*,
	image: str,
	task_dir: Path,
	work: Path,
	model: str,
	timeout_sec: int,
) -> tuple[str | None, float]:
	"""Run one fresh container. Returns ``(error, cost_usd)``."""
	environment = task_dir / "environment"
	workspace = work / "workspace"
	agent_dir = work / "agent"
	workspace.mkdir(parents=True, exist_ok=True)
	agent_dir.mkdir(parents=True, exist_ok=True)
	instruction = _instruction_text(task_dir / "instruction.md")
	container_config = "/opt/motifbench/mini_swe_agent.yaml"
	script = mini_swe_shell(
		model=model,
		instruction=shlex_quote(instruction),
		config_in_container=shlex_quote(container_config),
	)
	name = f"mb-mswe-{uuid.uuid4().hex[:12]}"
	argv = [
		"docker",
		"run",
		"--rm",
		"--name",
		name,
		"--cpus",
		"1",
		"--memory",
		"2048m",
		"-w",
		"/workspace",
		"-e",
		"OPENAI_API_KEY",
		"-e",
		"MINI_SWE_AGENT_MODEL_API_KEY_JSON",
		"-e",
		"MSWEA_CONFIGURED=true",
		"-e",
		"MSWEA_COST_TRACKING=ignore_errors",
		"-v",
		f"{CONFIG_PATH.resolve()}:{container_config}:ro",
		"-v",
		f"{(environment / 'macbeth.txt').resolve()}:/app/macbeth.txt:ro",
		"-v",
		f"{(environment / 'macbeth.pdf').resolve()}:/app/macbeth.pdf:ro",
		"-v",
		f"{(environment / 'motifs.json').resolve()}:/app/motifs.json:ro",
		"-v",
		f"{(environment / 'queries.json').resolve()}:/app/queries.json:ro",
		"-v",
		f"{workspace.resolve()}:/workspace",
		"-v",
		f"{agent_dir.resolve()}:/logs/agent",
		image,
		"bash",
		"-lc",
		script,
	]
	log_path = work / "mini-swe-agent.txt"
	run_env = os.environ.copy()
	api_key = run_env.get("OPENAI_API_KEY")
	if api_key:
		# Same channel Harbor uses: JSON text expanded by the container shell.
		run_env["MINI_SWE_AGENT_MODEL_API_KEY_JSON"] = json.dumps(api_key)
	try:
		completed = subprocess.run(
			argv,
			capture_output=True,
			text=True,
			timeout=timeout_sec,
			check=False,
			env=run_env,
		)
	except subprocess.TimeoutExpired as exc:
		subprocess.run(["docker", "kill", name], capture_output=True, check=False)
		stdout = exc.stdout if isinstance(exc.stdout, str) else ""
		stderr = exc.stderr if isinstance(exc.stderr, str) else ""
		log_path.write_text(scrub_text(stdout + "\n" + stderr), encoding="utf-8")
		_relax_tree(work)
		return f"timeout after {timeout_sec}s", _load_cost(agent_dir / TRAJECTORY_NAME)
	log_path.write_text(
		scrub_text((completed.stdout or "") + "\n" + (completed.stderr or "")),
		encoding="utf-8",
	)
	_relax_tree(work)
	scrub_tree(work)
	cost = _load_cost(agent_dir / TRAJECTORY_NAME)
	predictions = workspace / "predictions.json"
	if completed.returncode != 0:
		tail = scrub_text((completed.stderr or completed.stdout or "")[-500:])
		return f"mini-swe-agent rc={completed.returncode}: {tail}", cost
	if not predictions.is_file():
		return f"mini-swe-agent finished without writing {predictions}", cost
	return None, cost


def shlex_quote(value: str) -> str:
	"""Quote one shell word. Local copy so the command string is explicit."""
	return "'" + value.replace("'", "'\"'\"'") + "'"


def _execute_task(
	*,
	mode: str,
	task_dir: Path,
	trial_dir: Path,
	model: str,
	image: str,
	timeout_sec: int,
) -> dict[str, object]:
	work = trial_dir / task_dir.name
	staged = stage_task(task_dir, work)
	predictions_path = work / "workspace" / "predictions.json"
	error: str | None = None
	cost = 0.0
	if mode == "oracle":
		predictions_path.write_text(
			json.dumps(oracle_predictions(task_dir), indent=2) + "\n",
			encoding="utf-8",
		)
	else:
		try:
			error, cost = _run_container(
				image=image,
				task_dir=task_dir,
				work=work,
				model=model,
				timeout_sec=timeout_sec,
			)
		except (OSError, RuntimeError) as exc:
			error = scrub_text(str(exc))
	row: dict[str, object] = {
		"task_id": task_dir.name,
		"instruction_sha256": staged["instruction_sha256"],
		"cost_usd": cost,
		"error": error,
		"reward": None,
	}
	if error is None and predictions_path.is_file():
		row["reward"] = _score_task(task_dir, predictions_path, work)
	elif error is None:
		row["error"] = f"Missing predictions at {predictions_path}"
	return row


def run(argv: list[str] | None = None) -> dict[str, object]:
	"""Execute one parity invocation and return the summary object."""
	parser = argparse.ArgumentParser(
		description="MotifBench Harbor Scenario-2 mini-swe-agent parity runner."
	)
	parser.add_argument(
		"--mode",
		choices=("dry-run", "oracle", "mini-swe"),
		default="dry-run",
		help="dry-run and oracle make no paid API calls.",
	)
	parser.add_argument("--data-dir", type=Path, default=None)
	parser.add_argument(
		"--harbor-root",
		type=Path,
		default=None,
		help="Optional reference tree whose datasets/motifbench instructions are compared.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=_REPO_ROOT / "research_store" / "output" / "motifbench_harbor_parity",
	)
	parser.add_argument(
		"--granularity",
		choices=("scene-group", "act", "query"),
		default="scene-group",
	)
	parser.add_argument("--split", choices=("full", "parity"), default="parity")
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--task-ids", nargs="+", default=None)
	parser.add_argument("--runs", type=int, default=1)
	parser.add_argument("--model", default=DEFAULT_MODEL)
	parser.add_argument("--agent-version", default=AGENT_VERSION)
	parser.add_argument("--image", default=DEFAULT_IMAGE)
	parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
	parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
	parser.add_argument("--rebuild-image", action="store_true")
	parser.add_argument(
		"--compare-trajectories",
		nargs=2,
		type=Path,
		default=None,
		metavar=("LEFT", "RIGHT"),
		help="Compare the first system and user messages, then exit.",
	)
	parser.add_argument(
		"--scrub-tree",
		type=Path,
		default=None,
		help="Scrub secret-shaped strings under PATH and exit.",
	)
	args = parser.parse_args(argv)
	if args.compare_trajectories is not None:
		report = compare_prompt_prefixes(
			args.compare_trajectories[0], args.compare_trajectories[1]
		)
		print(json.dumps(report, indent=2))
		return report
	if args.scrub_tree is not None:
		count = scrub_tree(args.scrub_tree)
		print(f"scrubbed {count} secret-shaped sequences under {args.scrub_tree}")
		return {"scrubbed": count, "root": str(args.scrub_tree)}
	if args.runs < 1:
		parser.error("--runs must be >= 1")
	if args.concurrency < 1:
		parser.error("--concurrency must be >= 1")
	if args.mode == "mini-swe" and not os.environ.get("OPENAI_API_KEY"):
		parser.error("--mode mini-swe requires OPENAI_API_KEY in the environment.")

	data_dir = args.data_dir if args.data_dir is not None else env_data_dir()
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	tasks_root = output_dir / "tasks"
	if tasks_root.exists():
		_make_tree_writable(tasks_root)
		shutil.rmtree(tasks_root)
	builder = load_task_builder(
		output_dir=tasks_root,
		data_dir=data_dir,
		granularity=args.granularity,
		split=args.split,
		limit=args.limit,
		task_ids=args.task_ids,
	)
	builder.run()
	task_dirs = _generated_task_dirs(tasks_root)
	if not task_dirs:
		raise RuntimeError(f"Adapter generated no tasks in {tasks_root}")
	for task_dir in task_dirs:
		_check_instruction_identity(task_dir, args.harbor_root)

	command = displayed_mini_swe_command(args.model)
	if args.mode == "dry-run":
		staged_rows: list[dict[str, object]] = []
		stage_root = output_dir / "staged"
		if stage_root.exists():
			_make_tree_writable(stage_root)
			shutil.rmtree(stage_root)
		for task_dir in task_dirs:
			staged_rows.append(stage_task(task_dir, stage_root / task_dir.name))
		manifest: dict[str, object] = {
			"mode": "dry-run",
			"agent_invoked": False,
			"runs": args.runs,
			"model": args.model,
			"agent_version": args.agent_version,
			"granularity": args.granularity,
			"split": args.split,
			"n_tasks": len(staged_rows),
			"data_dir": None if data_dir is None else str(data_dir),
			"hf_repo": DEFAULT_HF_REPO,
			"hf_revision": DEFAULT_HF_REVISION,
			"harbor_root": None if args.harbor_root is None else str(args.harbor_root),
			"mini_swe_argv": command,
			"tasks": staged_rows,
		}
		(output_dir / "dry_run.json").write_text(
			json.dumps(manifest, indent=2) + "\n",
			encoding="utf-8",
		)
		print(
			f"dry-run staged {len(staged_rows)} tasks. "
			f"mini-swe-agent not invoked. Command would be: {command}"
		)
		return manifest

	if args.mode == "mini-swe":
		ensure_image(args.image, args.rebuild_image)

	per_run_micro: list[float] = []
	per_run_macro: list[float] = []
	trial_paths: list[str] = []
	last_metric: dict[str, float | int] = {}
	total_cost = 0.0
	total_errors = 0
	for run_index in range(1, args.runs + 1):
		trial_dir = output_dir / f"trial_{run_index}"
		if trial_dir.exists():
			_make_tree_writable(trial_dir)
			shutil.rmtree(trial_dir)
		trial_dir.mkdir(parents=True)
		rows_by_id: dict[str, dict[str, object]] = {}
		workers = 1 if args.mode == "oracle" else args.concurrency
		with ThreadPoolExecutor(max_workers=workers) as pool:
			futures = [
				pool.submit(
					_execute_task,
					mode=args.mode,
					task_dir=task_dir,
					trial_dir=trial_dir,
					model=args.model,
					image=args.image,
					timeout_sec=args.timeout_sec,
				)
				for task_dir in task_dirs
			]
			for future in as_completed(futures):
				row = future.result()
				task_id = row.get("task_id")
				if not isinstance(task_id, str):
					raise TypeError("task row is missing task_id")
				rows_by_id[task_id] = row
		task_rows = [rows_by_id[task_dir.name] for task_dir in task_dirs]
		run_cost = 0.0
		for row in task_rows:
			cost = row.get("cost_usd")
			if isinstance(cost, (int, float)) and not isinstance(cost, bool):
				run_cost += float(cost)
		total_cost += run_cost
		payload = _trial_payload(
			trial_index=run_index,
			mode=args.mode,
			model=None if args.mode == "oracle" else args.model,
			agent_version=None if args.mode == "oracle" else args.agent_version,
			granularity=args.granularity,
			split=args.split,
			task_rows=task_rows,
			agent_invoked=args.mode == "mini-swe",
			cost_usd=run_cost,
		)
		trial_path = output_dir / f"result_original_trial{run_index}.json"
		trial_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
		scrub_tree(trial_dir)
		scrub_tree(trial_path)
		trial_paths.append(str(trial_path))
		metrics = payload["metrics"]
		if not isinstance(metrics, dict):
			raise TypeError("trial metrics must be an object")
		errors = payload["n_errors"]
		if isinstance(errors, int):
			total_errors += errors
		last_metric = {
			str(key): value
			for key, value in metrics.items()
			if isinstance(value, (int, float)) and not isinstance(value, bool)
		}
		if "micro_hit_at_5" not in metrics:
			raise RuntimeError(
				f"Trial {run_index} did not produce micro Hit@5: {json.dumps(metrics)}"
			)
		per_run_micro.append(float(metrics["micro_hit_at_5"]))
		if "macro_hit_at_5" in metrics:
			per_run_macro.append(float(metrics["macro_hit_at_5"]))

	sem = sample_sem(per_run_micro)
	summary: dict[str, object] = {
		"metric": "Hit@5 (micro)",
		"mode": args.mode,
		"n_runs": len(per_run_micro),
		"n_tasks": len(task_dirs),
		"n_errors": total_errors,
		"total_queries": last_metric.get("total_queries"),
		"mean": sum(per_run_micro) / len(per_run_micro),
		"sample_sem": sem,
		"sample_sem_defined": sem is not None,
		"formatted": format_mean_sem(per_run_micro),
		"formatted_ascii": format_mean_sem_ascii(per_run_micro),
		"per_run_micro_hit_at_5": per_run_micro,
		"per_run_macro_hit_at_5": per_run_macro,
		"macro_formatted": format_mean_sem(per_run_macro) if per_run_macro else None,
		"agent_invoked": args.mode == "mini-swe",
		"model": None if args.mode == "oracle" else args.model,
		"agent_version": None if args.mode == "oracle" else args.agent_version,
		"cost_usd": total_cost,
		"trials": trial_paths,
	}
	(output_dir / "summary.json").write_text(
		json.dumps(summary, indent=2) + "\n",
		encoding="utf-8",
	)
	print(
		f"{args.mode} micro Hit@5 = {summary['formatted']} "
		f"over {summary['n_tasks']} tasks / {summary['total_queries']} queries "
		f"cost_usd={total_cost:.6f} errors={total_errors}"
	)
	return summary


def main(argv: list[str] | None = None) -> None:
	summary = run(argv)
	errors = summary.get("n_errors")
	if isinstance(errors, int) and errors:
		raise SystemExit(1)


if __name__ == "__main__":
	main()
