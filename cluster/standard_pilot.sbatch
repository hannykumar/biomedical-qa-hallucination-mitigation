#!/bin/bash
#SBATCH --job-name=biomed-c4-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --tmp=40G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/cluster/c4-pilot-%j.out
#SBATCH --error=outputs/cluster/c4-pilot-%j.err

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: sbatch cluster/standard_pilot.sbatch MODEL_KEY S1\|S2 RUN_ID CODE_REVISION" >&2
    exit 2
fi

MODEL_KEY="$1"
SETTING_ID="$2"
RUN_ID="$3"
CODE_REVISION="$4"

case "${MODEL_KEY}" in
    biomistral_7b|mistral_7b_instruct_v01) ;;
    *) echo "unsupported model key: ${MODEL_KEY}" >&2; exit 2 ;;
esac
case "${SETTING_ID}" in
    S1|S2) ;;
    *) echo "unsupported setting: ${SETTING_ID}" >&2; exit 2 ;;
esac
if [[ ! "${CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "CODE_REVISION must be a 40-character lowercase Git SHA" >&2
    exit 2
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/outputs/cluster"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biomed-hallucination

export HF_HOME="${SLURM_JOB_TMP}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
srun python -m src.generation.run_generation \
    --model-key "${MODEL_KEY}" \
    --setting-id "${SETTING_ID}" \
    --run-id "${RUN_ID}" \
    --code-revision "${CODE_REVISION}" \
    --cache-dir "${HF_HOME}" \
    --limit 5
