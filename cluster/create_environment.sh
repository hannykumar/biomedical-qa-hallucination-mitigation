#!/bin/bash

set -euo pipefail

PROJECT_DIR="${1:-${HOME}/biomed-hallucination}"
CONDA_ROOT="${HOME}/miniconda3"
ENV_NAME="biomed-hallucination"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "missing Miniconda initialization: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
    exit 2
fi

if [[ ! -f "${PROJECT_DIR}/environment-c3.yml" ]]; then
    echo "missing environment file: ${PROJECT_DIR}/environment-c3.yml" >&2
    exit 2
fi

if [[ ! -f "${PROJECT_DIR}/requirements-c3.txt" ]]; then
    echo "missing requirements file: ${PROJECT_DIR}/requirements-c3.txt" >&2
    exit 2
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "environment already exists: ${ENV_NAME}" >&2
    echo "remove or repair it explicitly before rerunning this script" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
conda env create --file environment-c3.yml

conda run --name "${ENV_NAME}" python -m pip install \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

conda run --name "${ENV_NAME}" python -m pip install \
    --requirement requirements-c3.txt

conda run --name "${ENV_NAME}" python -c \
    "import torch, transformers, datasets; print('python_environment=ok'); print('torch=' + torch.__version__); print('cuda_runtime=' + str(torch.version.cuda)); print('transformers=' + transformers.__version__); print('datasets=' + datasets.__version__)"
