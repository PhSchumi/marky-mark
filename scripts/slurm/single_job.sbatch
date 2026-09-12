#!/bin/bash
#SBATCH --job-name=wmbench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#
# Runs the whole benchmark (all variants, all datasets, all stages) in one GPU job.
# Prerequisite (once, on a node with internet):  python scripts/prepare_assets.py --model $MODEL --oracle $ORACLE
#
# Usage:  MODEL=facebook/opt-1.3b ORACLE=facebook/opt-2.7b RUN=opt13b sbatch scripts/slurm/single_job.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- environment: adapt to your cluster ---------------------------------------
# module load cuda/12.4 python/3.11
# source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WMBENCH_RESULTS="${WMBENCH_RESULTS:-$PWD/results}"

MODEL="${MODEL:-facebook/opt-1.3b}"
ORACLE="${ORACLE:-facebook/opt-2.7b}"
RUN="${RUN:-$(basename "$MODEL")}"

python scripts/run_benchmark.py --preset full \
  --model "$MODEL" --oracle "$ORACLE" --run-name "$RUN" \
  --stage all --no-progress "$@"
