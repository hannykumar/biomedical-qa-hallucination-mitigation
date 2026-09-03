#!/bin/bash
#SBATCH --job-name=biomed-c3-inspect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/cluster/c3-inspect-%j.out
#SBATCH --error=outputs/cluster/c3-inspect-%j.err

set -euo pipefail

mkdir -p outputs/cluster

echo "job_id=${SLURM_JOB_ID}"
echo "hostname=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx 'biomed-hallucination'; then
        conda activate biomed-hallucination
        python --version
        python - <<'PY'
import json

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the allocated GPU job")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("the allocated GPU does not support BF16")

device = torch.device("cuda:0")
matrix_size = 512
left = torch.ones((matrix_size, matrix_size), device=device, dtype=torch.bfloat16)
right = torch.ones((matrix_size, matrix_size), device=device, dtype=torch.bfloat16)
result = left @ right
torch.cuda.synchronize(device)

result_00 = float(result[0, 0].item())
checksum = float(result.float().sum().item())
expected_result_00 = float(matrix_size)
expected_checksum = float(matrix_size ** 3)
passed = result_00 == expected_result_00 and checksum == expected_checksum

record = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "gpu_name": torch.cuda.get_device_name(device),
    "matrix_dtype": str(result.dtype),
    "matrix_shape": list(result.shape),
    "result_00": result_00,
    "expected_result_00": expected_result_00,
    "checksum": checksum,
    "expected_checksum": expected_checksum,
    "gpu_compute_passed": passed,
}
print(json.dumps(record, sort_keys=True))

if not passed:
    raise RuntimeError("GPU matrix multiplication produced an unexpected result")
PY
    else
        echo "conda_environment=missing"
    fi
else
    echo "miniconda=missing"
fi
