#!/bin/bash
#SBATCH --job-name=biomed-c4-suite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --tmp=40G
#SBATCH --time=05:00:00
#SBATCH --output=outputs/cluster/c4-suite-%j.out
#SBATCH --error=outputs/cluster/c4-suite-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sbatch cluster/standard_smoke_suite.sbatch RUN_PREFIX CODE_REVISION" >&2
    exit 2
fi

RUN_PREFIX="$1"
CODE_REVISION="$2"
[[ "${RUN_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || { echo "invalid RUN_PREFIX" >&2; exit 2; }
[[ "${CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid CODE_REVISION" >&2; exit 2; }

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/outputs/cluster"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biomed-hallucination

export HF_HOME="${SLURM_JOB_TMP}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
for MODEL_KEY in mistral_7b_instruct_v01 biomistral_7b; do
    for SETTING_ID in S1 S2; do
        srun python -m src.generation.run_generation \
            --model-key "${MODEL_KEY}" \
            --setting-id "${SETTING_ID}" \
            --run-id "${RUN_PREFIX}-${MODEL_KEY}-${SETTING_ID}" \
            --code-revision "${CODE_REVISION}" \
            --cache-dir "${HF_HOME}" \
            --limit 5
    done
done

python -m src.generation.validate_smoke_suite \
    --run-prefix "${RUN_PREFIX}" \
    --summary-path "outputs/cluster/${RUN_PREFIX}.summary.json"
