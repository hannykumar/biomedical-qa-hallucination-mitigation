# PubMedQA Labeled Dataset Inspection

- Generated: `2026-08-11T13:12:49.117451+00:00`
- Dataset: `qiaojin/PubMedQA`
- Subset/split: `pqa_labeled` / `train`
- Resolved revision: `9001f2853fb87cab8d220904e0de81ac6973b318`
- Hugging Face fingerprint: `4d3b267f850a2785`
- Processed rows: **1000**
- Unique sample IDs: **1000**
- CSV SHA-256: `f9b12d498b0be9a9a5cfed2dff030af49501ddb9d5198da0077d21c1b1ceec2c`

## Final-decision distribution

| Label | Count |
|---|---:|
| maybe | 110 |
| no | 338 |
| yes | 552 |

## Context checks

| Measure | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Abstract sections per example | 1 | 3.358 | 9 |
| Flattened context characters | 294 | 1343.622 | 2729 |

## Normalization performed

- Stable sample IDs are derived from the source PubMed ID.
- Leading and trailing whitespace is removed from text fields.
- Final decisions are lowercased and validated against yes/no/maybe.
- Context sections retain source order and are joined with one blank line.
- Original context sections, section labels, and MeSH terms remain in JSON columns.
- No train/validation/test split is created during C1.

The processed CSV and metadata manifest are generated artifacts and are excluded from Git. Rebuild them with `python3 -m src.data.load_pubmedqa`.
