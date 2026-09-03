# CITEC GPU Cluster Setup for C3

This guide translates the university's official cluster instructions into the
steps required by this repository. Commands should be run from your own machine
or on the cluster as indicated. Never place a password or access token in this
repository.

The project owner requires explicit permission before every command that
submits, starts, or attaches to a GPU job. Do not run either `.sbatch` file merely
because it exists. Login-node access and environment setup do not grant GPU-job
permission.

Official references:

- [First steps](https://citec-gpu-cluster.pages.ub.uni-bielefeld.de/documentation/getting-started/first-steps/)
- [Slurm](https://citec-gpu-cluster.pages.ub.uni-bielefeld.de/documentation/getting-started/slurm/)
- [Cluster nodes](https://citec-gpu-cluster.pages.ub.uni-bielefeld.de/documentation/getting-started/nodes/)
- [Testing Torch](https://citec-gpu-cluster.pages.ub.uni-bielefeld.de/documentation/examples/torch/)
- [Compute-local storage](https://citec-gpu-cluster.pages.ub.uni-bielefeld.de/documentation/examples/localstorage/)

## 1. Access prerequisites

The cluster is reachable only from the Faculty of Technology network, through
the `citec` network, the faculty VPN, or the public proxy. The cluster account
must belong to the `gpuv2` UNIX group. Check access on a TechFak machine with:

```bash
id
```

The output must include `gpuv2`.

From inside the faculty network or VPN, log in with the TechFak username:

```bash
ssh TECHFAK_LOGIN@login-1.gpu.cit-ec.net
```

The documentation contains both `.net` and `.de` spellings for the login node;
the SSH example uses `.net`, so this project starts with that hostname and will
confirm it interactively.

The login node is only for file transfer, environment management, submitting
jobs, and monitoring them. Do not load the 7B models or run inference there.

## 2. Copy the repository

The repository is currently uncommitted, so copy it with `rsync` after login:

```bash
rsync -av --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'AGENTS.md' --exclude 'decision.md' --exclude 'handover.md' \
  --exclude 'flow.md' --exclude 'PROJECT_PLAN.md' --exclude 'PROJECT_START.md' \
  "/local/path/PubMedQA Hallucination Mitigation Project/" \
  TECHFAK_LOGIN@login-1.gpu.cit-ec.net:/homes/TECHFAK_LOGIN/biomed-hallucination/
```

The official docs warn that cluster homes and `/vol` are not backed up. Copy
important results back to another system after experiments.

## 3. Install Miniconda on the login node

Skip this section if `~/miniconda3/bin/conda` already exists.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh \
  -O ~/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh
echo "a6f98e6e19d5b7897ae887cd6af931eb863459f86ffd1a09cc370124cab0993e  $HOME/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh" \
  | sha256sum --check --strict
bash ~/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda config --set auto_activate_base false
```

Start a fresh shell, then create the pinned project environment:

```bash
cd ~/biomed-hallucination
conda config --file ~/miniconda3/.condarc --remove channels defaults
conda config --file ~/miniconda3/.condarc --add channels conda-forge
conda config --file ~/miniconda3/.condarc --prepend channels nvidia
conda config --file ~/miniconda3/.condarc --prepend channels pytorch
conda config --file ~/miniconda3/.condarc --set channel_priority strict
chmod u+x cluster/create_environment.sh
./cluster/create_environment.sh
conda activate biomed-hallucination
python -m unittest discover -s tests -v
```

On the installed Miniconda 26.7.1 client, `nodefaults` did not prevent the
Terms-of-Service plugin from seeing `defaults` in `~/miniconda3/.condarc`, while
`conda env create` rejected per-command `--override-channels` and `--channel`
arguments. This Miniconda installation was created only for the project, so its
root configuration replaces `defaults` with the three approved channels. No
Anaconda default-channel terms are accepted.

The initial Conda-based PyTorch solve remained unsatisfiable under strict
priority and tried to select an unintended CUDA 12.6 build. PyTorch 2.6.0 does
not have the required official Conda package; `cluster/create_environment.sh`
therefore creates a conda-forge Python/pip base, installs PyTorch from its
official CUDA 12.4 wheel index, and installs the remaining pins from
`requirements-c3.txt`. The actual driver, CUDA, and BF16 checks remain inside
the separately approved A40 inspection job.

## 4. Inspect an allocated A40 GPU

C3 targets one A40 in the `gpu` partition. Its documented 46 GB VRAM supports
the unquantized 7B BF16 model with substantially more margin than the 11–16 GB
cards in `study`.

```bash
cd ~/biomed-hallucination
mkdir -p outputs/cluster
sbatch cluster/inspect_cluster.sbatch
squeue -u "$USER"
```

When the job finishes, inspect:

```bash
cat outputs/cluster/c3-inspect-JOB_ID.out
cat outputs/cluster/c3-inspect-JOB_ID.err
```

The output must show an A40, CUDA available, and BF16 supported.

Verified on 2026-09-03: the approved inspection completed in 11 seconds with
exit code 0. It reported an NVIDIA A40 with 46,068 MiB, PyTorch 2.6.0+cu124,
CUDA runtime 12.4, BF16 support, and `gpu_compute_passed: true` for the
deterministic 512×512 BF16 matrix multiplication.

## 5. Run one-sample model smoke tests

The two approved repositories are public at their pinned revisions; an HF token
is not currently required. Each job downloads into `$SLURM_JOB_TMP`, which is
fast compute-local storage deleted after the job.

```bash
sbatch cluster/model_smoke.sbatch mistral_7b_instruct_v01
sbatch cluster/model_smoke.sbatch biomistral_7b
squeue -u "$USER"
```

After completion, verify both output/error pairs under `outputs/cluster/`. A C3
smoke test is successful only if it records the pinned model/tokenizer revisions,
an A40 device, BF16 dtype, package versions, prompt provenance, and a nonempty
completion.

Verified for Mistral-7B-Instruct-v0.1 on 2026-09-03: the approved smoke job
completed in 8 minutes 44 seconds with exit code 0. It loaded pinned revision
`ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca` in BF16 on an NVIDIA A40 and
recorded full prompt provenance plus a nonempty deterministic 16-token
completion.

Verified for BioMistral on 2026-09-03: the approved smoke job completed in 7
minutes 45 seconds with exit code 0. It loaded pinned revision
`9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5` in BF16 on an NVIDIA A40 and
recorded full prompt provenance plus a nonempty deterministic 16-token
completion. C3 cluster/model acceptance is complete.

## 6. Monitoring and cleanup

Useful commands from the official Slurm guide:

```bash
sinfo
squeue -u "$USER"
scontrol show job JOB_ID -d
srun --jobid=JOB_ID nvidia-smi
scancel JOB_ID
```

Do not use `scancel` unless the exact job ID has been checked.
