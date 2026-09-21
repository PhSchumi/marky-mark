#!/bin/bash
#SBATCH --job-name=wmbench
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=slurm.out
#
# Self-contained run of the whole benchmark: environment, assets and all four stages in one GPU job.
# Upload the repo to the cluster, check the CHANGE_ME lines below (repo path, scratch path) and
# TORCH_INDEX_URL in section 4, then:
#
#     sbatch scripts/slurm/all_in_one.sbatch
#
# Assumes compute nodes have internet access (verified: huggingface.co and github.com reachable).
# Every step is idempotent and every stage is resumable: if the job hits the time limit, submit the
# same file again and it continues from the cached records instead of starting over.

set -euo pipefail

# --- 1. repo root ------------------------------------------------------------------
# sbatch copies this script to the spool directory, so the repo path cannot be derived from $0,
# and SLURM_SUBMIT_DIR is wherever you happened to run `sbatch` (often scripts/slurm/).
# CHANGE_ME: the checkout on the cluster.
WMREPO="${WMREPO:-$HOME/ulm_project/marky-mark}"
cd "$WMREPO"
if [ ! -f pyproject.toml ]; then
  echo "WMREPO=$WMREPO is not a marky-mark checkout" >&2
  exit 1
fi

# --- 2. python ---------------------------------------------------------------------
# No `module load`: this cluster has no module command in batch shells, and the pip torch wheels
# ship their own CUDA runtime, so only the NVIDIA driver on the compute node is needed.
# python3 comes from the environment sbatch inherited from the submit shell (conda base).
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), f"need python >= 3.10, got {sys.version}"'
echo "=== python: $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="

# --- 3. paths ----------------------------------------------------------------------
# A directory with >= 40 GB free: the HF cache is ~8 GB (opt-1.3b + opt-2.7b), the SIR assets are
# 1.3 GB and the venv is ~6 GB. /scratch is not writable on this cluster, so the beegfs home is
# the default; override with `WMROOT=/path sbatch ...` if there is a designated work directory.
export WMROOT="${WMROOT:-$HOME/wmbench}"

export HF_HOME="$WMROOT/hf_cache"
export WMBENCH_THIRD_PARTY="$WMROOT/third_party"   # pinned MarkLLM checkout (datasets ship inside it)
export WMBENCH_ARTIFACTS="$WMROOT/artifacts"       # SIR transform network + embedder
export WMBENCH_RESULTS="$WMROOT/results"
export NLTK_DATA="$WMROOT/nltk_data"
mkdir -p "$WMROOT"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-facebook/opt-1.3b}"
ORACLE="${ORACLE:-facebook/opt-2.7b}"
RUN="${RUN:-opt13b}"

# --- 4. virtualenv -----------------------------------------------------------------
# Created on the first run, reused afterwards. It lives next to the caches on scratch so the
# ~6 GB of torch wheels do not land in the home quota.
# TORCH_INDEX_URL must match a CUDA version the node driver supports.
VENV="${VENV:-$WMROOT/venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
if [ ! -d "$VENV" ]; then
  echo "=== creating $VENV ==="
  "$PYTHON" -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install torch --index-url "$TORCH_INDEX_URL"
  pip install -r requirements.txt
else
  source "$VENV/bin/activate"
fi

# --- 5. fail fast before spending GPU hours ----------------------------------------
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see a GPU: check the driver and the torch CUDA build"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

# --- 6. assets (MarkLLM at the pinned commit, SIR weights, NLTK data, model weights) -
echo "=== preparing assets ==="
python scripts/prepare_assets.py --model "$MODEL" --oracle "$ORACLE"

# --- 7. the benchmark --------------------------------------------------------------
# --device cuda rather than the default "auto": auto silently falls back to CPU, and CPU records
# cannot be mixed with CUDA ones (MarkLLM seeds torch.Generator(device=...)), which would only
# surface at the detect stage after the generations were already paid for.
echo "=== running benchmark: $MODEL (oracle $ORACLE) -> $WMBENCH_RESULTS/$RUN ==="
python scripts/run_benchmark.py --preset full \
  --model "$MODEL" --oracle "$ORACLE" --run-name "$RUN" \
  --stage all --device cuda \
  --results-root "$WMBENCH_RESULTS" --no-progress "$@"

echo "=== done: $WMBENCH_RESULTS/$RUN/summary.md ==="
