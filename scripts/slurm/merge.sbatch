#!/bin/bash
#SBATCH --job-name=wmbench-merge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#
# Job B (after array_generate.sbatch): cross-variant stages on one GPU.
#   detect_none : every watermark detector scores the `none` generations
#   quality     : PPL with the oracle (generator is freed first), diversity, BLEU, ROUGE, pass@1
#   report      : tables and plots (pure pandas)

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
ORACLE="${ORACLE:-facebook/opt-2.7b}"
RUN="${RUN:-$(basename "$MODEL")}"

python scripts/run_benchmark.py --preset full \
  --model "$MODEL" --oracle "$ORACLE" --run-name "$RUN" \
  --stage detect_none,quality,report --no-progress "$@"
