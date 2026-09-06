# Biomedical QA Hallucination Mitigation

This repository studies whether standard decoding, DoLA, CAD, and a later hybrid method can reduce hallucinated biomedical claims in PubMedQA explanations.

C1-C3 establish the pinned PubMedQA data contract, versioned prompts, and a
verified university-A40 runtime for BioMistral-7B and
Mistral-7B-Instruct-v0.1. C4 implements resumable S1/S2 standard generation.
Its four-combination smoke suite completed, but the original minimal prompt
failed the label-format and output-length readiness gate. Prompt version 2 now
requires an explicit final label and no more than three explanation sentences.
C7 parses raw generations into separate, auditable derivative files.

## C1 dataset setup

The dataset build requires Python plus Hugging Face `datasets` and
`huggingface_hub`. Dependency versions will be pinned after the university
environment is inspected.

From the repository root, build the complete pinned dataset:

```bash
python3 -m src.data.load_pubmedqa
```

Run the dependency-light unit tests:

```bash
python3 -m unittest discover -s tests -v
```

The build produces:

- `data/processed/pubmedqa_labeled_clean.csv`
- `data/processed/pubmedqa_labeled_clean.metadata.json`
- `reports/dataset_inspection.md`

The CSV and manifest are reproducible generated artifacts and are ignored by
Git. The inspection report is versioned.

For a limited smoke build, provide separate paths for all three artifacts;
the command rejects limited runs that could overwrite the canonical full-data
outputs.

## C2 prompt setup

Load the validated prompt configuration once, then build either approved prompt
type from a normalized C1 example:

```python
from src.prompts.build_prompts import build_prompt, load_prompt_config

config = load_prompt_config()
built = build_prompt(
    example,
    prompt_type="question_context",
    config=config,
)

print(built.text)
print(built.metadata())
```

`BuiltPrompt` contains exact text plus the prompt type, version, format,
template SHA-256, and rendered-prompt SHA-256. C2 intentionally does not apply
a tokenizer chat template; that model-specific serialization is decided during
C3/C4.

## C3 model setup

The cluster-specific procedure is documented in
[reports/university_gpu_setup.md](reports/university_gpu_setup.md). C3 targets
one A40 in the Slurm `gpu` partition, uses the pinned environment in
`environment-c3.yml`, and keeps Hugging Face model caches in compute-local
`$SLURM_JOB_TMP`.

Local dependency-light validation:

```bash
python3 -m unittest discover -s tests -v
bash -n cluster/inspect_cluster.sbatch cluster/model_smoke.sbatch
```

On the cluster, after creating the environment and `outputs/cluster/`:

```bash
sbatch cluster/inspect_cluster.sbatch
sbatch cluster/model_smoke.sbatch mistral_7b_instruct_v01
sbatch cluster/model_smoke.sbatch biomistral_7b
```

C3 is complete: both pinned model revisions loaded in BF16 on allocated A40
GPUs and produced nonempty deterministic smoke completions.

## C4 standard baseline generation

`src/generation/run_generation.py` reads the normalized C1 CSV, resolves S1 or
S2, builds the C2 prompt, calls `src/decoding/standard.py`, and appends one raw
JSONL record per completed sample. Re-running the same immutable run skips
completed sample IDs and rejects changed manifests or duplicate records.

The pilot script is intentionally limited to five examples. It must be
submitted only after explicit approval for that GPU action:

```bash
sbatch cluster/standard_pilot.sbatch MODEL_KEY S1\|S2 RUN_ID CODE_REVISION
```

Model loading and generation are CUDA-only. Local tests use fakes and never
load either model.

### Automated baseline smoke suite

From macOS, one command runs four C4 pilots sequentially on one A40: both fixed
models with S1 and S2. It synchronizes only Git-tracked files,
reuses one temporary password-authenticated SSH connection, runs dependency-light
checks, polls Slurm, and downloads logs, JSONL, manifests, and a readiness summary.
The sample limit and maximum token count are explicit command arguments and are
recorded in each run manifest.
The Slurm job has a two-hour safety ceiling; this is separate from Codex account
usage and actual GPU time stops when the job finishes. The temporary connection
closes when the command exits, and results remain Git-ignored under `outputs/`:

```bash
./cluster/run_standard_smoke_suite_remote.sh \
  USER@login-1.gpu.cit-ec.net \
  /homes/USER/biomed-hallucination \
  UNIQUE_RUN_PREFIX \
  SAMPLE_LIMIT \
  MAX_NEW_TOKENS
```

Running this command submits one GPU job containing all four pilots. Obtain
approval for that specific suite before invoking it. Enter the password only at
the terminal's hidden SSH prompt; the script never stores it or creates
persistent passwordless access. If the local connection is interrupted, Slurm
continues and keeps completed artifacts on university storage. Rerun the same
command and run prefix after reconnecting; the saved job ID resumes polling and
collection without submitting another job. The downloaded `RUN_PREFIX.sacct.txt`
records actual GPU-job elapsed time. After collection, C7 parses the downloaded
records and writes a grouped parser summary automatically.

## C7 output parsing

Parse one or more raw C4 JSONL files without changing them:

```bash
python3 -m src.generation.parse_outputs \
  outputs/generations/standard/RUN_PREFIX-*.jsonl \
  --summary-path outputs/generations/parsed/RUN_PREFIX.parser-summary.json
```

Parsed records retain every raw field and add the extracted label, explanation,
parser status/errors, parser version, and exact/normalized/noncompliant format
classification. The summary reports failures separately for each model and
setting. This step is CPU-only.
