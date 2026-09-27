#!/bin/bash
#SBATCH --job-name=wmbench-robust
#SBATCH --partition=day
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm.out
#
# Self-contained run of the whole benchmark INCLUDING the robustness track (word deletion + DIPPER)
# in one GPU job. DIPPER is T5-XXL (~22 GB in bf16), which needs a >= 40 GB card: the L40S (48 GB)
# fits it. Check the CHANGE_ME lines below (repo path, work path) and TORCH_INDEX_URL in section 4,
# then:
#
#     sbatch scripts/slurm/all_in_one_robustness.sbatch
#
# Assumes compute nodes have internet access (verified: huggingface.co and github.com reachable).
# Every step is idempotent and every stage is resumable: if the job hits the time limit, submit the
# same file again and it continues the unfinished run from the cached records instead of starting
# over. Once that run has finished, the next submission starts a clean run under a new name
# (opt13b-robust-<timestamp>). `FRESH=1 sbatch ...` forces a new run even if the last one is
# unfinished. The DIPPER paraphrase of 500 x 6 C4 texts alone takes a few hours, so a resume may be
# needed on `day`.

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
# A directory with >= 65 GB free: the HF cache is ~8 GB (opt-1.3b + opt-2.7b) plus ~22 GB for the
# DIPPER T5-XXL weights, the SIR assets are 1.3 GB and the venv is ~6 GB. /scratch is not writable
# on this cluster, so the beegfs home is the default; override with `WMROOT=/path sbatch ...`.
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

# Run name. The name of the last run started by this file is kept in $RUN_STATE:
#   - that run is unfinished (no summary.md yet)  -> resume it
#   - it finished, there is none, or FRESH=1      -> start a clean run under a new timestamped name
# `RUN=<name> sbatch ...` bypasses this and uses <name> as given.
RUN_STATE="$WMROOT/robustness_run_name"
if [ -z "${RUN:-}" ]; then
  if [ "${FRESH:-0}" != "1" ] && [ -f "$RUN_STATE" ]; then
    LAST_RUN="$(cat "$RUN_STATE")"
    if [ ! -f "$WMBENCH_RESULTS/$LAST_RUN/summary.md" ]; then
      RUN="$LAST_RUN"
      echo "=== resuming unfinished run $RUN ==="
    fi
  fi
  if [ -z "${RUN:-}" ]; then
    RUN="opt13b-robust-$(date +%Y%m%d-%H%M%S)"
    while [ -e "$WMBENCH_RESULTS/$RUN" ]; do
      sleep 1
      RUN="opt13b-robust-$(date +%Y%m%d-%H%M%S)"
    done
    echo "$RUN" > "$RUN_STATE"
    echo "=== starting new run $RUN ==="
  fi
fi

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

# protobuf lets transformers extract DIPPER's T5-v1.1 SentencePiece tokenizer (spiece.model);
# without it transformers falls back to a TikToken path that then needs tiktoken and fails.
# Idempotent so it also fixes an already-created venv on resume.
pip install --quiet protobuf

# --- 5. fail fast before spending GPU hours ----------------------------------------
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see a GPU: check the driver and the torch CUDA build"
assert torch.cuda.is_bf16_supported(), "DIPPER (T5-XXL) overflows in fp16 and needs bf16"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

# --- 6. assets: MarkLLM, SIR weights, NLTK, model weights AND the DIPPER paraphraser -
# --dipper also fetches the DIPPER tokenizer and the attack similarity model.
echo "=== preparing assets (with DIPPER) ==="
python scripts/prepare_assets.py --model "$MODEL" --oracle "$ORACLE" --dipper

# --- 7. the benchmark, robustness track included -----------------------------------
# --robustness adds stage `attack` (word deletion + DIPPER on the length-controlled watermarked C4
# texts) and the attacked:<name> detection rows. --device cuda rather than the default "auto":
# auto silently falls back to CPU, and CPU records cannot be mixed with CUDA ones (MarkLLM seeds
# torch.Generator(device=...)), which would only surface at the detect stage after the generations
# were already paid for.
echo "=== running benchmark + robustness: $MODEL (oracle $ORACLE) -> $WMBENCH_RESULTS/$RUN ==="
python scripts/run_benchmark.py --preset full \
  --model "$MODEL" --oracle "$ORACLE" --run-name "$RUN" \
  --stage all --robustness --device cuda \
  --results-root "$WMBENCH_RESULTS" --no-progress "$@"

echo "=== done: $WMBENCH_RESULTS/$RUN/summary.md ==="
