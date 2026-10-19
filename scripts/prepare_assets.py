#!/usr/bin/env python
"""Prefetch everything the benchmark needs so compute nodes can run offline.

Run this on a node with internet access (login node / laptop) with the same
``HF_HOME`` that the jobs will use::

    export HF_HOME=/shared/hf_cache
    python scripts/prepare_assets.py --model facebook/opt-1.3b --oracle facebook/opt-2.7b [--semstamp]

Then in the job scripts::

    export HF_HOME=/shared/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

What is fetched:
  * MarkLLM at the pinned commit -> third_party/MarkLLM (datasets ship inside it)
  * GSM8K test split (OpenAI repository, pinned commit) -> artifacts/gsm8k/test.jsonl
  * SIR transform network + compositional-BERT embedder -> artifacts/sir/
  * NLTK punkt / punkt_tab (SemStamp)
  * generator, oracle and (optionally) the SemStamp sentence embedder -> HF cache
  * with --dipper: the DIPPER paraphraser (T5-XXL, ~22 GB), its tokenizer and the attack
    similarity model (stage `attack` of the robustness track)
  * artifacts/assets.json with the resolved Hub revisions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmbench import bootstrap  # noqa: E402


def _revision(repo_id: str) -> str | None:
    if Path(repo_id).is_dir():
        return None  # local checkpoint
    try:
        from huggingface_hub import model_info

        return model_info(repo_id).sha
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="facebook/opt-1.3b", help="generator model id or local path")
    p.add_argument("--oracle", default="facebook/opt-2.7b", help="PPL oracle id ('none' to skip)")
    p.add_argument("--semstamp", action="store_true", help="also fetch the SemStamp sentence embedder")
    p.add_argument("--dipper", action="store_true",
                   help="also fetch the DIPPER paraphraser, its tokenizer and the attack similarity model")
    p.add_argument("--skip-models", action="store_true", help="only MarkLLM, SIR assets and NLTK data")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("prepare_assets")

    root = bootstrap.ensure_markllm()
    log.info("MarkLLM ready at %s (commit %s)", root, bootstrap.MARKLLM_COMMIT[:8])
    for rel in ("dataset/c4/processed_c4.json", "dataset/wmt16_de_en/validation.jsonl",
                "dataset/cnn_dailymail/test-00000-of-00001.jsonl", "dataset/human_eval/test.jsonl"):
        assert (root / rel).exists(), f"missing dataset file {root / rel}"
    gsm8k = bootstrap.ensure_gsm8k()
    log.info("GSM8K test split: %s", gsm8k)
    sir = bootstrap.ensure_sir_assets()
    log.info("SIR assets: %s", sir)
    bootstrap.ensure_nltk()

    assets = {"markllm_commit": bootstrap.MARKLLM_COMMIT, "sir": sir,
              "sir_transform_revision": _revision(bootstrap.SIR_TRANSFORM_REPO),
              "sir_embedder_revision": _revision(bootstrap.SIR_EMBEDDER_REPO),
              "gsm8k": {"path": str(gsm8k), "commit": bootstrap.GSM8K_COMMIT, "sha1": bootstrap.file_sha1(gsm8k)},
              "models": {}}
    if not args.skip_models:
        names = [args.model] + ([] if args.oracle.lower() in ("none", "") else [args.oracle])
        bootstrap.prefetch_models(names)
        for n in names:
            assets["models"][n] = _revision(n)
        if args.semstamp:
            bootstrap.ensure_semstamp_assets()
            assets["models"][bootstrap.SEMSTAMP_EMBEDDER_REPO] = _revision(bootstrap.SEMSTAMP_EMBEDDER_REPO)
        if args.dipper:
            from wmbench.config import BenchmarkConfig

            cfg = BenchmarkConfig()
            names = [cfg.dipper_model, cfg.dipper_tokenizer] + ([cfg.attack_similarity_model] if cfg.attack_similarity_model else [])
            bootstrap.prefetch_models(names)
            assets["attacks"] = {n: _revision(n) for n in names}
    out = bootstrap.default_artifacts_dir() / "assets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(assets, indent=2))
    log.info("wrote %s", out)
    log.info("compute nodes can now run with HF_HUB_OFFLINE=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
