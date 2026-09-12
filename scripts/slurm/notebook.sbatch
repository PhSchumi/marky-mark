#!/bin/bash
#SBATCH --job-name=wmbench-nb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#
# Executes the notebook headlessly on a GPU node and saves the executed copy with outputs.
# The notebook reads its configuration from WMBENCH_* environment variables.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# module load cuda/12.4 python/3.11
# source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

export WMBENCH_SMOKE_TEST="${WMBENCH_SMOKE_TEST:-0}"
export WMBENCH_MODEL="${WMBENCH_MODEL:-facebook/opt-1.3b}"
export WMBENCH_ORACLE="${WMBENCH_ORACLE:-facebook/opt-2.7b}"
export WMBENCH_RUN_NAME="${WMBENCH_RUN_NAME:-nb-$(basename "$WMBENCH_MODEL")}"

# KERNEL: name of the Jupyter kernel that has this repo's dependencies (python -m ipykernel install --user --name wmbench)
KERNEL="${KERNEL:-wmbench}"
export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-$PWD/results/.jupyter}"   # isolate from user-level nbconvert extensions
mkdir -p "$JUPYTER_CONFIG_DIR"

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name="$KERNEL" \
  --output "executed_${WMBENCH_RUN_NAME}.ipynb" \
  notebooks/watermark_benchmark.ipynb
