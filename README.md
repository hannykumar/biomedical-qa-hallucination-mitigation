# Biomedical QA Hallucination Mitigation

This project tests whether different decoding methods can make biomedical
question answering more accurate and better grounded in the supplied medical
evidence.

## Project in one minute

For each PubMedQA example, we have:

- a biomedical question;
- a research context that may help answer it;
- a gold label: `yes`, `no`, or `maybe`; and
- a gold long answer written from the source evidence.

We ask two 7B language models to answer the same examples, first without the
context and then with it. Later, we will repeat the comparison with DoLA, CAD,
and a possible hybrid method.

```text
PubMedQA question and optional context
        ↓
Model generates a yes/no/maybe label and a short explanation
        ↓
C7 safely extracts the label and explanation
        ↓
Evaluation compares them with the gold label and gold long answer
```

The two fixed models are:

- `BioMistral/BioMistral-7B`
- `mistralai/Mistral-7B-Instruct-v0.1`

## What one test means

The latest smoke test used 25 questions in four combinations:

| Model | S1: question only | S2: question + context |
|---|---:|---:|
| Mistral | 25 outputs | 25 outputs |
| BioMistral | 25 outputs | 25 outputs |
| **Total** | **50 outputs** | **50 outputs** |

Therefore, 25 questions produced 100 outputs. The future full standard
baseline will use all 1,000 questions and produce 4,000 outputs:
`1,000 questions × 2 models × 2 settings`.

## Experiment roadmap

| ID | Model input | Decoding method | Status |
|---|---|---|---|
| S1 | Question only | Standard | Smoke-tested |
| S2 | Question + context | Standard | Smoke-tested |
| S3 | Question only | DoLA | Planned |
| S4 | Question + context | DoLA | Planned |
| S5 | Context contrasted with no context | CAD | Planned |
| S6 | Question + context | CAD + DoLA-inspired hybrid | Later |

We complete and evaluate the standard S1/S2 baseline before implementing DoLA
or CAD. This gives those later methods a trustworthy comparison point.

## Prompt evolution

The answer prompt was improved through three small GPU tests.

### Version 1 — minimal instruction

```text
Question: {question}

Answer the question and explain your reasoning briefly.
```

This allowed too much freedom. Only 8 of 20 outputs started with a usable
answer label, and 4 reached the 256-token ceiling.

### Version 2 — structured answer

```text
Question: {question}

Your first line must be exactly one of:
Final answer: yes
Final answer: no
Final answer: maybe
Your second line must begin with "Explanation:" and contain no more than three concise sentences.
```

This improved label formatting, but BioMistral still omitted required content
and many answers became too long.

### Version 3 — short, mandatory answer

Version 3 is the selected baseline prompt. The question-only form is:

```text
Question: {question}

Respond with exactly two lines and no other text.
Line 1 must be exactly one of:
Final answer: yes
Final answer: no
Final answer: maybe
Line 2 must begin with "Explanation:" and contain exactly one concise sentence.
Always include both lines. Do not stop after line 1. Do not begin with "Explanation:".
```

S2 uses the same instruction but adds `Context: {context}` before the question.
The authoritative current templates are in
[`configs/prompts.yaml`](configs/prompts.yaml).

## Prompt smoke-test results

| Prompt | Questions | Outputs | Labels found | Missing-content failures | Exact two-line format | Token-ceiling hits |
|---|---:|---:|---:|---:|---:|---:|
| v1, minimal | 5 | 20 | 8/20 prefix signal | Not yet measured by C7 | Not measured | 4/20 at 256 tokens |
| v2, structured | 25 | 100 | 92/100 | 15/100 | 27/100 | 18/100 at 128 tokens |
| **v3, concise** | **25** | **100** | **100/100** | **1/100** | **50/100** | **2/100 at 120 tokens** |

Versions 2 and 3 used the same 25 questions, so their results are directly
comparable. Version 3 reduced missing-content failures from 15% to 1% and
token-ceiling hits from 18% to 2%.

We accept the observed 1% missing-content rate and will use prompt version 3
for the full baseline candidate. The incomplete record will remain visible in
the results; it will not be guessed, repaired, or silently removed. The two
120-token ceiling hits are also reported separately. The stored full-baseline
default remains 128 tokens.

### Important: formatting is not accuracy

`100/100 labels found` does **not** mean the model achieved 100% accuracy. It
only means C7 could read a `yes`, `no`, or `maybe` label from every output.

Accuracy will be calculated later by comparing each parsed model label with
the dataset's gold label. Explanation quality will be compared with the gold
long answer using ROUGE-L, BERTScore, cosine similarity, and output length.
These similarity scores are useful comparison signals, but they are not direct
proof that an explanation is free of hallucinations.

## Current status

- C1 dataset preparation: complete for all 1,000 examples.
- C2 prompt construction: complete; version 3 selected.
- C3 model and university A40 setup: complete for both models.
- C4 standard generation: complete for all 4,000 baseline outputs.
- C7 output parser: complete for the full baseline.
- C8 final-label accuracy: complete for all 4,000 baseline outputs.
- C9 ROUGE-L and output-length evaluation: complete for all 4,000 outputs.
- C10 BERTScore and cosine-similarity evaluation: complete for 3,995 available explanations.
- Combined Phase 1 MVP results table: next.
- DoLA and CAD: planned after the baseline is reproducible and evaluated.

### Full baseline generation result

| Model and setting | Raw outputs | Parsed labels | Missing-content failures | 128-token ceiling hits |
|---|---:|---:|---:|---:|
| Mistral S1 | 1,000 | 1,000 | 0 | 0 |
| Mistral S2 | 1,000 | 1,000 | 0 | 6 |
| BioMistral S1 | 1,000 | 1,000 | 0 | 34 |
| BioMistral S2 | 1,000 | 999 | 5 | 25 |
| **Total** | **4,000** | **3,999** | **5** | **65** |

### Full baseline label accuracy

| Model and setting | Correct | Accuracy | Parser failures |
|---|---:|---:|---:|
| Mistral S1 | 475/1,000 | 47.5% | 0 |
| Mistral S2 | 687/1,000 | 68.7% | 0 |
| BioMistral S1 | 548/1,000 | 54.8% | 0 |
| BioMistral S2 | 708/1,000 | 70.8% | 5 |

`unknown` labels count as incorrect. Parser failures are reported separately and
do not change the 1,000-example denominator.

### Full baseline ROUGE-L and output length

| Model and setting | Mean ROUGE-L | Explanations scored | Mean generated tokens |
|---|---:|---:|---:|
| Mistral S1 | 0.1953 | 1,000/1,000 | 42.034 |
| Mistral S2 | 0.2213 | 1,000/1,000 | 50.572 |
| BioMistral S1 | 0.1965 | 1,000/1,000 | 48.350 |
| BioMistral S2 | 0.2110 | 995/1,000 | 48.112 |

ROUGE-L here is case-insensitive word-level F1 without stemming. It measures
lexical overlap with the gold long answer, not medical correctness or
hallucination. Five missing BioMistral S2 explanations remain null and are
excluded from the mean rather than guessed or silently scored.

### Full baseline semantic similarity

| Model and setting | Mean BERTScore F1 | Mean cosine similarity | Explanations scored |
|---|---:|---:|---:|
| Mistral S1 | 0.6627 | 0.9507 | 1,000/1,000 |
| Mistral S2 | 0.6700 | 0.9530 | 1,000/1,000 |
| BioMistral S1 | 0.6590 | 0.9469 | 1,000/1,000 |
| BioMistral S2 | 0.6611 | 0.9474 | 995/1,000 |

These are within-pipeline similarity signals, not percentages and not direct
proof of factual correctness or freedom from hallucination. The five missing
BioMistral S2 explanations remain null. No scored text reached either encoder's
input limit.

All four GPU generation steps completed in 1 hour 49 minutes. Slurm marked the
wrapper job as failed only because the first validator split at a valid Unicode
line-separator character inside biomedical text. The raw files were not
damaged: all 4,000 JSON records validate, and a streaming validator fix now has
test coverage. The strict zero-failure readiness flag remains false, while the
observed 0.125% overall missing-content rate remains visible for evaluation.

Every new GPU action still requires specific approval before submission.

## Running the project

### Build the dataset and run local tests

The dataset build requires Python, Hugging Face `datasets`, and
`huggingface_hub`:

```bash
python3 -m src.data.load_pubmedqa
python3 -m unittest discover -s tests -v
```

The build creates these Git-ignored reproducible artifacts:

- `data/processed/pubmedqa_labeled_clean.csv`
- `data/processed/pubmedqa_labeled_clean.metadata.json`
- `reports/dataset_inspection.md`

A limited dataset smoke build must use separate output paths so it cannot
replace the complete 1,000-example artifacts.

### Build a prompt

```python
from src.prompts.build_prompts import build_prompt, load_prompt_config

config = load_prompt_config()
built = build_prompt(example, prompt_type="question_context", config=config)

print(built.text)
print(built.metadata())
```

Every built prompt records its type, version, format, template SHA-256, and
rendered-prompt SHA-256.

### University GPU setup

The cluster procedure is documented in
[`reports/university_gpu_setup.md`](reports/university_gpu_setup.md). Both
pinned models have successfully loaded in BF16 and generated text on an A40.
Model loading and generation are CUDA-only; local tests use fakes.

### Run the automated baseline smoke suite

From macOS, this command synchronizes committed files, runs both models under
S1 and S2 sequentially in one Slurm job, validates the results, and downloads
the raw and parsed artifacts:

```bash
./cluster/run_standard_smoke_suite_remote.sh \
  USER@login-1.gpu.cit-ec.net \
  /homes/USER/biomed-hallucination \
  UNIQUE_RUN_PREFIX \
  SAMPLE_LIMIT \
  MAX_NEW_TOKENS \
  WALLTIME
```

`WALLTIME` is optional and defaults to `02:00:00` for smoke tests. The prepared
full baseline uses `SAMPLE_LIMIT=1000`, `MAX_NEW_TOKENS=128`, and
`WALLTIME=18:00:00`, producing 4,000 outputs sequentially on one A40.

Obtain explicit approval for that specific GPU suite before running it. The
password is entered only at the terminal's hidden SSH prompt and is never
stored. If SSH disconnects, Slurm continues; rerunning the same command with
the same prefix resumes monitoring and collection without submitting a second
job.

### Parse raw outputs on CPU

```bash
python3 -m src.generation.parse_outputs \
  outputs/generations/standard/RUN_PREFIX-*.jsonl \
  --summary-path outputs/generations/parsed/RUN_PREFIX.parser-summary.json
```

C7 preserves every raw field and writes separate parsed JSONL files containing
the extracted label, explanation, parser status, errors, and format class. Raw
model outputs are never overwritten.

### Evaluate final-label accuracy on CPU

```bash
python3 -m src.evaluation.accuracy \
  outputs/generations/parsed/RUN_PREFIX-*.jsonl \
  --metrics-path outputs/metrics/RUN_PREFIX-label-accuracy.jsonl \
  --table-path outputs/tables/RUN_PREFIX-label-accuracy.csv
```

C8 rejects unaligned run order, counts `unknown` as incorrect, and writes
Git-ignored per-sample scores plus a model/setting aggregate table.

### Evaluate ROUGE-L and output length on CPU

```bash
python3 -m src.evaluation.overlap_metrics \
  outputs/generations/parsed/RUN_PREFIX-*.jsonl \
  --metrics-path outputs/metrics/RUN_PREFIX-rouge-l-length.jsonl \
  --table-path outputs/tables/RUN_PREFIX-rouge-l-length.csv
```

C9 reuses C8's paired-order validation and writes per-sample scores plus grouped
means, population standard deviations, missing counts, and parser-failure counts.

### Evaluate BERTScore and cosine similarity on CPU

```bash
conda create --prefix .venv --channel conda-forge --override-channels \
  python=3.11 pip=25.1
.venv/bin/python -m pip install -r requirements-evaluation.txt
.venv/bin/python -m src.evaluation.semantic_metrics \
  outputs/generations/parsed/RUN_PREFIX-*.jsonl \
  --metrics-path outputs/metrics/RUN_PREFIX-semantic.jsonl \
  --table-path outputs/tables/RUN_PREFIX-semantic.csv
```

C10 enforces the pinned package and model revisions in
`configs/evaluation.yaml`, records that provenance in every result, and leaves
missing explanations unscored.
