#!/bin/bash
#SBATCH --job-name=wmbench-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%A_%a.out
#
# Job A: one array task per variant runs stage "generate" (generation + self-detection on
# its own and the human texts). Tasks write only to their own <variant>.jsonl files; the shared
# manifest.json is updated under a lock. Submit job B (merge.sbatch) with --dependency=afterok:<job id>.
#
# Usage:
#   JOBA=$(MODEL=facebook/opt-1.3b RUN=opt13b sbatch --parsable scripts/slurm/array_generate.sbatch)
#   MODEL=facebook/opt-1.3b ORACLE=facebook/opt-2.7b RUN=opt13b sbatch --dependency=afterok:$JOBA scripts/slurm/merge.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# module load cuda/12.4 python/3.11
# source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WMBENCH_RESULTS="${WMBENCH_RESULTS:-$PWD/results}"

MODEL="${MODEL:-facebook/opt-1.3b}"
RUN="${RUN:-$(basename "$MODEL")}"
# Edit to add SemStamp (slow) as a 7th task and set --array=0-6.
VARIANTS=(none KGW MorphMark PF SynthID SIR)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

echo "task $SLURM_ARRAY_TASK_ID -> variant $VARIANT"
python scripts/run_benchmark.py --preset full \
  --model "$MODEL" --run-name "$RUN" \
  --stage generate --variants "$VARIANT" --results-root "$WMBENCH_RESULTS" --no-progress "$@"
