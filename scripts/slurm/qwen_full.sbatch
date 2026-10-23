#!/bin/bash
#SBATCH --job-name=wmbench-qwen
#SBATCH --partition=L40Sday
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-qwen-%j.out
#SBATCH --signal=B:USR1@900
#
# The whole Qwen study in one submission:
#   generator Qwen/Qwen2.5-1.5B-Instruct, perplexity oracle Qwen/Qwen2.5-7B-Instruct (both bf16),
#   watermarks KGW, MorphMark, PF, SynthID, SIR + the unwatermarked baseline,
#   tasks C4 (detection, perplexity, diversity), WMT16 (BLEU), CNN/DailyMail (ROUGE),
#   GSM8K (all 1,319 test problems, final-answer accuracy; GSM8K_REPEATS answers per problem),
#   robustness: word deletion + DIPPER paraphrase on the C4 texts,
#   and the quality degradation vs. the unwatermarked model (degradation.csv, summary.md).
#
# Submit from the repo root on the cluster:
#
#     git pull && sbatch scripts/slurm/qwen_full.sbatch
#
# With one answer per GSM8K problem the run takes roughly 15 h (the 500-problem run took 8.3 h, of which
# GSM8K was about 4 h). Every extra answer per problem adds about 10.5 h (GSM8K_REPEATS=3: about 36 h).
# 15 minutes before the 20 h limit the job resubmits itself
# (at most MAX_RESUBMITS times) and the new job resumes from the cached records, so one submission
# is enough. When everything is done the results are packed into
#     $WMROOT/results/<run>.tar.gz          (default WMROOT: ~/wmbench)
# and the download command is printed at the end of slurm-qwen-<jobid>.out.
#
# `FRESH=1 sbatch ...` starts a new run even if the last one is unfinished;
# `RUN=<name> sbatch ...` resumes or starts the run <name>.
# `GSM8K_REPEATS=3 sbatch ...` asks every GSM8K problem 3 times (own seed each); the report then
# scores each problem by its share of correct answers. `GSM8K_N=500` reproduces the first Qwen run.

set -euo pipefail

# --- 1. repo root ------------------------------------------------------------------
# sbatch copies this script to the spool directory, so the repo path cannot be derived from $0.
# CHANGE_ME if the checkout lives elsewhere on the cluster.
WMREPO="${WMREPO:-$HOME/ulm_project/marky-mark}"
cd "$WMREPO"
if [ ! -f pyproject.toml ]; then
  echo "WMREPO=$WMREPO is not a marky-mark checkout" >&2
  exit 1
fi
RESUBMITS="${RESUBMITS:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-3}"

# --- 2. python ---------------------------------------------------------------------
# python3 comes from the environment sbatch inherited from the submit shell (conda base).
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), f"need python >= 3.10, got {sys.version}"'
echo "=== python: $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="

# --- 3. paths ----------------------------------------------------------------------
# Needs ~90 GB free: HF cache ~3 GB (Qwen 1.5B) + ~15 GB (Qwen 7B) + ~22 GB (DIPPER T5-XXL),
# SIR assets 1.3 GB, venv ~6 GB. Shares the caches and venv with the OPT runs.
export WMROOT="${WMROOT:-$HOME/wmbench}"

export HF_HOME="$WMROOT/hf_cache"
export WMBENCH_THIRD_PARTY="$WMROOT/third_party"   # pinned MarkLLM checkout (datasets ship inside it)
export WMBENCH_ARTIFACTS="$WMROOT/artifacts"       # SIR assets, GSM8K test split
export WMBENCH_RESULTS="$WMROOT/results"
export NLTK_DATA="$WMROOT/nltk_data"
mkdir -p "$WMROOT" "$WMBENCH_RESULTS"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ORACLE="${ORACLE:-Qwen/Qwen2.5-7B-Instruct}"
GSM8K_N="${GSM8K_N:-1319}"            # 1319 = the whole GSM8K test split
GSM8K_REPEATS="${GSM8K_REPEATS:-1}"    # answers per GSM8K problem

# Run name. The last run started by this file is kept in $RUN_STATE:
#   - unfinished (no summary.md yet)        -> resume it
#   - finished, none yet, or FRESH=1        -> start a new run qwen15b-<timestamp>
RUN_STATE="$WMROOT/qwen_run_name"
if [ -z "${RUN:-}" ]; then
  if [ "${FRESH:-0}" != "1" ] && [ -f "$RUN_STATE" ]; then
    LAST_RUN="$(cat "$RUN_STATE")"
    if [ ! -f "$WMBENCH_RESULTS/$LAST_RUN/summary.md" ]; then
      RUN="$LAST_RUN"
      echo "=== resuming unfinished run $RUN ==="
    fi
  fi
  if [ -z "${RUN:-}" ]; then
    RUN="qwen15b-$(date +%Y%m%d-%H%M%S)"
    echo "$RUN" > "$RUN_STATE"
    echo "=== starting new run $RUN ==="
  fi
fi

# --- 4. virtualenv -----------------------------------------------------------------
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
# protobuf: needed for DIPPER's T5 SentencePiece tokenizer (see all_in_one_robustness.sbatch).
pip install --quiet protobuf

# --- 5. fail fast before spending GPU hours ----------------------------------------
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see a GPU: check the driver and the torch CUDA build"
assert torch.cuda.is_bf16_supported(), "Qwen2.5 and DIPPER run in bf16"
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

# --- 6. assets: MarkLLM, GSM8K, SIR weights, NLTK, Qwen models, DIPPER --------------
echo "=== preparing assets ==="
python scripts/prepare_assets.py --model "$MODEL" --oracle "$ORACLE" --dipper

# --- 7. the benchmark ----------------------------------------------------------------
# Runs in the background so the USR1 signal (15 min before the time limit) reaches the trap below,
# which stops python and resubmits this file for the same run.
resubmit() {
  echo "=== time limit close: stopping and resubmitting run $RUN ==="
  kill -TERM "$PY_PID" 2>/dev/null || true
  wait "$PY_PID" 2>/dev/null || true
  if [ "$RESUBMITS" -lt "$MAX_RESUBMITS" ]; then
    sbatch --export=ALL,RUN="$RUN",RESUBMITS=$((RESUBMITS + 1)) "$WMREPO/scripts/slurm/qwen_full.sbatch"
  else
    echo "=== reached MAX_RESUBMITS=$MAX_RESUBMITS; resume by hand: RUN=$RUN sbatch scripts/slurm/qwen_full.sbatch ==="
  fi
  exit 0
}
trap resubmit USR1

echo "=== running benchmark: $MODEL (oracle $ORACLE) -> $WMBENCH_RESULTS/$RUN (resubmit $RESUBMITS) ==="
echo "=== GSM8K: $GSM8K_N problems x $GSM8K_REPEATS answer(s) ==="
python scripts/run_benchmark.py --preset full \
  --model "$MODEL" --oracle "$ORACLE" --run-name "$RUN" \
  --dtype bfloat16 --device cuda \
  --datasets c4,wmt16_de_en,cnn_dailymail,gsm8k \
  --n-samples "{\"gsm8k\": $GSM8K_N}" --max-new-tokens '{"gsm8k": 512}' \
  --gsm8k-repeats "$GSM8K_REPEATS" \
  --stage all --robustness \
  --results-root "$WMBENCH_RESULTS" --no-progress "$@" &
PY_PID=$!
wait "$PY_PID"

# --- 8. package the results for download ---------------------------------------------
mkdir -p "$WMBENCH_RESULTS/$RUN/logs"
cp "$SLURM_SUBMIT_DIR"/slurm-qwen-*.out "$WMBENCH_RESULTS/$RUN/logs/" 2>/dev/null || true
cp "$WMREPO"/slurm-qwen-*.out "$WMBENCH_RESULTS/$RUN/logs/" 2>/dev/null || true
tar czf "$WMBENCH_RESULTS/$RUN.tar.gz" -C "$WMBENCH_RESULTS" "$RUN"

echo "=== done: $WMBENCH_RESULTS/$RUN/summary.md ==="
echo "=== quality degradation: $WMBENCH_RESULTS/$RUN/degradation.csv ==="
echo "=== download from your laptop (replace <cluster> with your login host):"
echo "    scp <cluster>:$WMBENCH_RESULTS/$RUN.tar.gz run_results/ && tar xzf run_results/$RUN.tar.gz -C run_results/"
