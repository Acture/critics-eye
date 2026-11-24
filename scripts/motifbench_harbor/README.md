# MotifBench Harbor Parity (`Acture/critics-eye@harbor-parity`)

Standalone upstream Scenario-2 harness for the MotifBench Harbor adapter ([harbor-index/adapters#35](https://github.com/harbor-index/adapters/pull/35)). It does not import a Harbor checkout. Both sides run `mini-swe-agent@2.2.8` with `openai/gpt-5.4-mini` through LiteLLM chat completions (`openai_chat_completions`) with identical YAML configuration and byte-identical rendered `instruction.md` prompts.

## Shared Setup

- Dataset pin: [`Acture/MotifBench`](https://huggingface.co/datasets/Acture/MotifBench) @ `12fadc6a46aa8d2fcc7848777574122afeccf59f`
- Tasks: 12 contiguous Act-respecting scene-group tasks covering 270 cleaned query lines and 218 candidate motifs
- Read-only workspace inputs: `/app/macbeth.txt`, `/app/macbeth.pdf`, `/app/motifs.json`, `/app/queries.json`
- Agent image toolchain: `python:3.13-slim`, `pypdf==7.1.1`, `uv tool install --python 3.12 mini-swe-agent==2.2.8 --with 'litellm[mcp]' --with orjson --with fastapi`
- Config: `mini_swe_agent.yaml` (`step_limit: 40`, `cost_limit: 1.0`, `environment.timeout: 60`, `drop_params: true`), layered on `-c mini`
- Scoring: `evaluate.py` (`Hit@5` micro & macro, `Hit@1`, `Hit@10`, `MRR@20`, `nDCG@10`)

`OPENAI_API_KEY` is read from the process environment and passed to Docker via `-e OPENAI_API_KEY`; it is never written into the image, config files, or logs.

## Oracle & Dry-Run Verification (No Paid API Calls)

When `--data-dir` is omitted, `run_parity_mini_swe_agent.py` automatically downloads the pinned revision from Hugging Face (`Acture/MotifBench@12fadc6a46aa8d2fcc7848777574122afeccf59f`):

```bash
python3 scripts/motifbench_harbor/run_parity_mini_swe_agent.py \
  --mode oracle \
  --granularity scene-group \
  --split parity \
  --runs 1 \
  --output-dir /tmp/motifbench-oracle
```

`--mode dry-run` stages `/app` and prints the exact `mini-swe-agent` command without starting Docker or calling the API. Run the parity unit tests with:

```bash
pytest --noconftest tests/test_motifbench_harbor_parity.py
```

## Paid Original-Side Parity Run

Set `OPENAI_API_KEY` in the environment, then run:

```bash
python3 scripts/motifbench_harbor/run_parity_mini_swe_agent.py \
  --mode mini-swe \
  --model openai/gpt-5.4-mini \
  --agent-version 2.2.8 \
  --timeout-sec 1200 \
  --concurrency 4 \
  --runs 1 \
  --output-dir ./parity_out/run1
```

Repeat for `run2` and `run3`. Each task runs in a fresh container executing:

```text
mini-swe-agent --yolo --model=openai/gpt-5.4-mini --task=<instruction> \
  --output=/logs/agent/mini-swe-agent.trajectory.json --cost-limit 1.0 \
  -c mini -c model.model_kwargs.timeout=3600 -c mini_swe_agent.yaml \
  -c "model.model_kwargs.api_key=$MINI_SWE_AGENT_MODEL_API_KEY_JSON" \
  -c model.model_kwargs.custom_llm_provider=openai \
  -c model.model_class=litellm --exit-immediately
```

## Harbor-Side Parity Run

From the `harbor-index/adapters` repository root (after `cd src/motifbench && uv run motifbench --output-dir ../../datasets/motifbench`):

```bash
harbor run -c src/motifbench/run_motifbench_parity.yaml --job-name motifbench-parity-run1
harbor run -c src/motifbench/run_motifbench_parity.yaml --job-name motifbench-parity-run2
harbor run -c src/motifbench/run_motifbench_parity.yaml --job-name motifbench-parity-run3
```

## Measured Results (2026-11-23)

3 runs per side, 12 tasks (270 queries), `mini-swe-agent==2.2.8` + `openai/gpt-5.4-mini`. Sample SEM is `stdev / sqrt(3)`.

| Side | Hit@5 (micro) | Hit@5 (macro) | USD (3 runs) |
| --- | --- | --- | --- |
| Original | 0.2901 +/- 0.0193 | 0.2977 +/- 0.0197 | 0.68481615 |
| Harbor | 0.2593 +/- 0.0119 | 0.2663 +/- 0.0095 | 0.53542530 |

- Original micro runs: `0.32222222222222224`, `0.25555555555555554`, `0.29259259259259257`
- Harbor micro runs: `0.2777777777777778`, `0.23703703703703705`, `0.26296296296296295`
- Run ranges overlap on both micro and macro Hit@5.
- Offline rescore max absolute difference across all 36 Harbor `predictions.json` files vs `reward.json`: `0.0`.
