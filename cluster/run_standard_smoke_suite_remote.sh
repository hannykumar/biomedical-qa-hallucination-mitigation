#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: $0 USER@HOST REMOTE_PROJECT_DIR RUN_PREFIX" >&2
}

if [[ ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "run this script from the local Mac" >&2
    exit 2
fi

TARGET="$1"
REMOTE_DIR="$2"
RUN_PREFIX="$3"
POLL_SECONDS="${POLL_SECONDS:-15}"

[[ "${TARGET}" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$ ]] || { echo "invalid USER@HOST" >&2; exit 2; }
[[ "${REMOTE_DIR}" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "REMOTE_PROJECT_DIR must be an absolute simple path" >&2; exit 2; }
[[ "${RUN_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || { echo "invalid RUN_PREFIX" >&2; exit 2; }
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

caffeinate -w $$ >/dev/null 2>&1 &
echo "Opening one temporary password-authenticated SSH session..."
ssh "${SSH_OPTIONS[@]}" -M -S "${CONTROL_SOCKET}" -o ControlPersist=no -Nf "${TARGET}"

echo "Synchronizing Git-tracked files only..."
git ls-files -z | rsync -avR --from0 --files-from=- \
    -e "ssh -S ${CONTROL_SOCKET}" ./ "${TARGET}:${REMOTE_DIR}/"

echo "Running dependency-light cluster checks..."
ssh -S "${CONTROL_SOCKET}" "${TARGET}" \
    "cd '${REMOTE_DIR}' && source \"\${HOME}/miniconda3/etc/profile.d/conda.sh\" && conda activate biomed-hallucination && python -m unittest discover -s tests -v && bash -n cluster/standard_smoke_suite.sbatch"

echo "Submitting the approved four-run GPU smoke suite..."
JOB_ID="$(ssh -S "${CONTROL_SOCKET}" "${TARGET}" \
    "cd '${REMOTE_DIR}' && mkdir -p outputs/cluster && sbatch --parsable cluster/standard_smoke_suite.sbatch '${RUN_PREFIX}' '${CODE_REVISION}'")"
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || { echo "unexpected Slurm job ID: ${JOB_ID}" >&2; exit 1; }
echo "JOB_ID=${JOB_ID}"

while STATUS="$(ssh -S "${CONTROL_SOCKET}" "${TARGET}" "squeue -h -j '${JOB_ID}' -o '%T %M %R'")" && [[ -n "${STATUS}" ]]; do
    echo "${STATUS}"
    sleep "${POLL_SECONDS}"
done

mkdir -p outputs/cluster outputs/generations/standard
ACCOUNTING="$(ssh -S "${CONTROL_SOCKET}" "${TARGET}" \
    "sacct -j '${JOB_ID}' --format=JobID,State,ExitCode,Elapsed,NodeList")"
printf '%s\n' "${ACCOUNTING}" | tee "outputs/cluster/${RUN_PREFIX}.sacct.txt"
STATE="$(ssh -S "${CONTROL_SOCKET}" "${TARGET}" \
    "sacct -X -n -j '${JOB_ID}' --format=State | head -n 1 | xargs")"
rsync -av -e "ssh -S ${CONTROL_SOCKET}" \
    "${TARGET}:${REMOTE_DIR}/outputs/cluster/c4-suite-${JOB_ID}.out" \
    "${TARGET}:${REMOTE_DIR}/outputs/cluster/c4-suite-${JOB_ID}.err" \
    outputs/cluster/
[[ "${STATE%%+*}" == "COMPLETED" ]] || { echo "suite finished with state ${STATE}; generation files remain on the cluster" >&2; exit 1; }

for MODEL_KEY in mistral_7b_instruct_v01 biomistral_7b; do
    for SETTING_ID in S1 S2; do
        RUN_ID="${RUN_PREFIX}-${MODEL_KEY}-${SETTING_ID}"
        rsync -av -e "ssh -S ${CONTROL_SOCKET}" \
            "${TARGET}:${REMOTE_DIR}/outputs/generations/standard/${RUN_ID}.jsonl" \
            "${TARGET}:${REMOTE_DIR}/outputs/generations/standard/${RUN_ID}.manifest.json" \
            outputs/generations/standard/
    done
done
rsync -av -e "ssh -S ${CONTROL_SOCKET}" \
    "${TARGET}:${REMOTE_DIR}/outputs/cluster/${RUN_PREFIX}.summary.json" \
    outputs/cluster/

shasum -a 256 \
    outputs/generations/standard/"${RUN_PREFIX}"-* \
    "outputs/cluster/${RUN_PREFIX}.summary.json" \
    "outputs/cluster/${RUN_PREFIX}.sacct.txt" \
    "outputs/cluster/c4-suite-${JOB_ID}.out" \
    "outputs/cluster/c4-suite-${JOB_ID}.err"
echo "CLUSTER_SMOKE_SUITE_COMPLETE job_id=${JOB_ID} summary=outputs/cluster/${RUN_PREFIX}.summary.json"
