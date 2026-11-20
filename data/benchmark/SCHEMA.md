# Benchmark format

One JSON object per line (`.jsonl`, UTF-8). Real file: `data/benchmark/benchmark.jsonl` (to be written with the sericulture officer). `EXAMPLE_benchmark.jsonl` only shows the shape; its rows are placeholders, not data.

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique id, e.g. `q001` |
| `question_kn` | yes | Question in Kannada as a farmer would ask it |
| `question_en` | no | English version (helps analysis, not used by the system) |
| `topic` | yes | `mulberry`, `rearing`, `disease_pest`, `cocoon_market`, `schemes`, `other` |
| `label` | yes | `answer` (sources contain the answer) or `abstain` (correct behaviour is to refuse, e.g. not in documents, needs field inspection) |
| `gold_answer_kn` | if label=answer | Expert-verified reference answer |
| `gold_answer_en` | no | English reference answer |
| `source_doc` | if label=answer | PDF filename that supports the answer |
| `source_page` | no | Page number in that PDF |
| `verified_by` | yes | Role or initials of the verifying expert (no full names needed) |
| `difficulty` | no | `easy`, `medium`, `hard` |
| `split` | yes | `dev` (threshold tuning) or `test` (final numbers only) |
| `style` | no | `script` (Kannada script) or `mixed` (Kannada-English mix) |

Validate with: `backend/.venv/bin/python eval/experiments/validate_benchmark.py data/benchmark/benchmark.jsonl`
