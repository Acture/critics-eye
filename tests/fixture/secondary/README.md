# Omitted fixtures

Three files that some tests reference are intentionally **not included** in
this artifact:

- Two PDFs — full-text copies of two copyrighted criticism sources
  (Bloom & Marson (eds.), *Macbeth* — Bloom's Shakespeare Through the Ages,
  2008; Sachon, *Shakespeare, Objects and Phenomenology*, 2020).
- `mac_reviews.jsonl` — extraction-parser test fixture whose records include
  full-sentence quotes from several copyrighted criticism sources.

See the top-level `NOTICE` for the full rationale. Tests depending on these
(`tests/conftest.py`'s `mac_reviews`/`secondary_pdfs` fixtures and their
consumers) will need substitute local files with the same schema to run.
