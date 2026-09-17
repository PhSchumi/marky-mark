#!/bin/bash
#SBATCH --job-name=wmbench-attack
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#
# Robustness track, between array_generate.sbatch and merge.sbatch: stage "attack" paraphrases the
# length-controlled watermarked C4 texts of every variant with DIPPER (T5-XXL, ~22 GB in bf16:
# needs a 40 GB+ GPU) and applies word deletion. Prefetch once on a node with internet:
#   python scripts/prepare_assets.py --model $MODEL --oracle $ORACLE --dipper
# Submit merge.sbatch with --robustness afterwards so stage detect scores the attacked texts.
#
# Usage:
#   JOBA=$(MODEL=facebook/opt-1.3b RUN=opt13b sbatch --parsable scripts/slurm/array_generate.sbatch)
#   JOBB=$(MODEL=facebook/opt-1.3b RUN=opt13b sbatch --parsable --dependency=afterok:$JOBA scripts/slurm/attack.sbatch)
#   MODEL=facebook/opt-1.3b ORACLE=facebook/opt-2.7b RUN=opt13b \
#     sbatch --dependency=afterok:$JOBB scripts/slurm/merge.sbatch --robustness

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

python scripts/run_benchmark.py --preset full \
  --model "$MODEL" --run-name "$RUN" \
  --stage attack --robustness --results-root "$WMBENCH_RESULTS" --no-progress "$@"
