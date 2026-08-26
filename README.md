# MotifBench — extraction pipeline

Code release for **"Critic's Eye: Review-Grounded Graph Reasoning for
Long-Form Motif Discovery"** (EMNLP 2026). This repository ships the
**pipeline that builds the MotifBench dataset** — TEI parsing and
criticism-derived motif extraction — not model training or evaluation code.
The dataset itself is released separately on the Hugging Face Hub:

> **Dataset**: <https://huggingface.co/datasets/Acture/MotifBench> (CC BY-NC 4.0)

## What this is

A **motif** is a recurring salient thematic element a work returns to —
blood in *Macbeth*, appearing across lines that share no vocabulary ("the
multitudinous seas incarnadine", "Out, damned spot!", "all the perfumes of
Arabia"). Recovering one means noticing that scattered, dissimilar passages
belong together, which is a different problem from long-context retrieval,
where the answer is somewhere in the input.

**MotifBench** operationalizes this — given a query line from *Macbeth*,
rank a fixed inventory of motifs — with labels drawn from published literary
criticism rather than our own annotation. This repository is the pipeline
that produces it: parsing the play into a narrative graph, then extracting
motif-bearing citations from criticism and resolving them onto that graph.

## Repository layout

| Path | Contents |
|---|---|
| `src/critics_eye/core/primary/` | TEI-XML parsing → narrative graph, BGE feature embedding, PyG `HeteroData` export |
| `src/critics_eye/core/secondary/` | Criticism extraction, motif clustering, graph injection |
| `src/critics_eye/utils/` | Shared utilities (embedding, PDF/text extraction, graph I/O) |
| `src/critics_eye/cli/` | `ce-cli` — the `primary` and `secondary` pipeline subcommands |
| `tests/` | Test suite (a few fixtures requiring separately-licensed source files are intentionally omitted — see `tests/fixture/*/README.md` and `NOTICE`) |
| `dataset/` | Git submodule pointing at the MotifBench dataset on the Hugging Face Hub |

## Quickstart

```fish
uv sync
uv run ce-cli --help
uv run pytest

git submodule update --init  # fetches dataset/ from Hugging Face
```

## License

AGPL-3.0 (see `LICENSE`). See `NOTICE` for test-fixture attributions and what is
intentionally omitted. The dataset has its own CC BY-NC 4.0 license — see the
dataset repository.

## Citation

```bibtex
@inproceedings{sun2026criticseye,
  title     = {Critic's Eye: Review-Grounded Graph Reasoning for Long-Form Motif Discovery},
  author    = {Sun, Xinyu and Yin, Zhangyue and Wang, Xin},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```
