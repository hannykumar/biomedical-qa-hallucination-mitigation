#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: $0 USER@HOST REMOTE_PROJECT_DIR MODEL_KEY S1|S2 RUN_ID" >&2
}

if [[ ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 5 ]]; then
    usage
    exit 2
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "run this script from the local Mac" >&2
    exit 2
fi

TARGET="$1"
REMOTE_DIR="$2"
MODEL_KEY="$3"
SETTING_ID="$4"
RUN_ID="$5"
POLL_SECONDS="${POLL_SECONDS:-15}"

[[ "${TARGET}" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$ ]] || { echo "invalid USER@HOST" >&2; exit 2; }
[[ "${REMOTE_DIR}" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "REMOTE_PROJECT_DIR must be an absolute simple path" >&2; exit 2; }
[[ "${MODEL_KEY}" == "mistral_7b_instruct_v01" || "${MODEL_KEY}" == "biomistral_7b" ]] || { echo "unsupported model key" >&2; exit 2; }
[[ "${SETTING_ID}" == "S1" || "${SETTING_ID}" == "S2" ]] || { echo "setting must be S1 or S2" >&2; exit 2; }
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid RUN_ID" >&2; exit 2; }
[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || { echo "POLL_SECONDS must be a positive integer" >&2; exit 2; }

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "commit tracked changes before running so the manifest code revision is truthful" >&2
    exit 2
fi

CODE_REVISION="$(git rev-parse HEAD)"
TEMP_DIR="$(mktemp -d)"
CONTROL_SOCKET="${TEMP_DIR}/ssh"
SSH_OPTIONS=(-o PubkeyAuthentication=no -o PreferredAuthentications=password)

cleanup() {
    ssh -S "${CONTROL_SOCKET}" -O exit "${TARGET}" >/dev/null 2>&1 || true
    rmdir "${TEMP_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Opening one temporary password-authenticated SSH session..."
ssh "${SSH_OPTIONS[@]}" -M -S "${CONTROL_SOCKET}" -o ControlPersist=no -Nf "${TARGET}"

echo "Synchronizing Git-tracked files only..."
git ls-files -z | rsync -avR --from0 --files-from=- \
    -e "ssh -S ${CONTROL_SOCKET}" ./ "${TARGET}:${REMOTE_DIR}/"

echo "Running dependency-light cluster checks..."
ssh -S "${CONTROL_SOCKET}" "${TARGET}" \
    "cd '${REMOTE_DIR}' && source \"\${HOME}/miniconda3/etc/profile.d/conda.sh\" && conda activate biomed-hallucination && python -m unittest discover -s tests -v && bash -n cluster/standard_pilot.sbatch"

echo "Submitting the approved GPU job..."
JOB_ID="$(ssh -S "${CONTROL_SOCKET}" "${TARGET}" \
    "cd '${REMOTE_DIR}' && mkdir -p outputs/cluster && sbatch --parsable cluster/standard_pilot.sbatch '${MODEL_KEY}' '${SETTING_ID}' '${RUN_ID}' '${CODE_REVISION}'")"
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || { echo "unexpected Slurm job ID: ${JOB_ID}" >&2; exit 1; }
echo "JOB_ID=${JOB_ID}"

while STATUS="$(ssh -S "${CONTROL_SOCKET}" "${TARGET}" "squeue -h -j '${JOB_ID}' -o '%T %M %R'")" && [[ -n "${STATUS}" ]]; do
    echo "${STATUS}"
    sleep "${POLL_SECONDS}"
done

ssh -S "${CONTROL_SOCKET}" "${TARGET}" \
    "sacct -j '${JOB_ID}' --format=JobID,State,ExitCode,Elapsed,NodeList"

mkdir -p outputs/cluster outputs/generations/standard
STATE="$(ssh -S "${CONTROL_SOCKET}" "${TARGET}" \
    "sacct -X -n -j '${JOB_ID}' --format=State | head -n 1 | xargs")"
rsync -av -e "ssh -S ${CONTROL_SOCKET}" \
    "${TARGET}:${REMOTE_DIR}/outputs/cluster/c4-pilot-${JOB_ID}.out" \
    "${TARGET}:${REMOTE_DIR}/outputs/cluster/c4-pilot-${JOB_ID}.err" \
    outputs/cluster/
[[ "${STATE%%+*}" == "COMPLETED" ]] || { echo "job finished with state ${STATE}" >&2; exit 1; }

rsync -av -e "ssh -S ${CONTROL_SOCKET}" \
    "${TARGET}:${REMOTE_DIR}/outputs/generations/standard/${RUN_ID}.jsonl" \
    "${TARGET}:${REMOTE_DIR}/outputs/generations/standard/${RUN_ID}.manifest.json" \
    outputs/generations/standard/

shasum -a 256 \
    "outputs/generations/standard/${RUN_ID}.jsonl" \
    "outputs/generations/standard/${RUN_ID}.manifest.json"
echo "CLUSTER_PILOT_COMPLETE job_id=${JOB_ID} run_id=${RUN_ID}"
