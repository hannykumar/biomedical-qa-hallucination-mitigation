#!/bin/bash
#SBATCH --job-name=biomed-c3-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --tmp=40G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/cluster/c3-smoke-%j.out
#SBATCH --error=outputs/cluster/c3-smoke-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch cluster/model_smoke.sbatch biomistral_7b|mistral_7b_instruct_v01" >&2
    exit 2
fi

MODEL_KEY="$1"
case "${MODEL_KEY}" in
    biomistral_7b|mistral_7b_instruct_v01) ;;
    *)
        echo "unsupported model key: ${MODEL_KEY}" >&2
        exit 2
        ;;
esac

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_DIR}/outputs/cluster"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biomed-hallucination

export HF_HOME="${SLURM_JOB_TMP}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
srun python -m src.models.smoke_test \
    --model-key "${MODEL_KEY}" \
    --cache-dir "${HF_HOME}" \
    --max-new-tokens 16
