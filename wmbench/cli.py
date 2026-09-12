"""Command-line entry point (``python -m wmbench.cli`` or ``scripts/run_benchmark.py``)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from wmbench.config import BenchmarkConfig, load_config_file
from wmbench.evaluate import STAGES, Runner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wmbench",
        description="Benchmark MarkLLM watermarks (KGW, MorphMark, PF, SynthID, SIR, SemStamp) on one HF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--preset", choices=["full", "smoke"], default=None,
                   help="start from a preset (default: WMBENCH_SMOKE_TEST env decides, else full)")
    p.add_argument("--config", type=Path, help="JSON/YAML file with BenchmarkConfig fields (overrides preset)")
    p.add_argument("--stage", default="all", help=f"comma list of stages from {STAGES} or 'all'")
    p.add_argument("--model", help="Hugging Face model id or path for the generator")
    p.add_argument("--oracle", help="PPL oracle model id; 'none' scores with the generator")
    p.add_argument("--run-name")
    p.add_argument("--variants", help="comma list, e.g. none,KGW,PF")
    p.add_argument("--datasets", help="comma list, e.g. c4,wmt16_de_en")
    p.add_argument("--n-samples", help="int applied to every dataset, or JSON object per dataset")
    p.add_argument("--max-new-tokens", help="int applied to every dataset, or JSON object per dataset")
    p.add_argument("--device", help="cuda | cuda:0 | mps | cpu | auto")
    p.add_argument("--dtype", help="auto | float16 | bfloat16 | float32")
    p.add_argument("--seed", type=int)
    p.add_argument("--results-root", help="directory that holds results/<run-name>")
    p.add_argument("--robustness", action="store_true", help="also score word-deletion-attacked texts")
    p.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--semstamp-n-max", type=int)
    p.add_argument("--no-resume", action="store_true", help="regenerate even if records exist")
    p.add_argument("--force-new-run", action="store_true", help="append to a results dir with a different config hash")
    p.add_argument("--allow-device-mismatch", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    return p


def _int_or_json(value: str, base: dict) -> dict:
    value = value.strip()
    if value.startswith("{"):
        return {**base, **json.loads(value)}
    return {k: int(value) for k in base}


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    if args.config:
        cfg = load_config_file(args.config)
    elif args.preset == "smoke":
        cfg = BenchmarkConfig.smoke_test()
    elif args.preset == "full":
        cfg = BenchmarkConfig.full()
    else:
        cfg = BenchmarkConfig.from_env()
    upd: dict = {}
    if args.model:
        upd["model_name"] = args.model
    if args.oracle is not None:
        upd["oracle_model_name"] = None if args.oracle.lower() in ("none", "null", "") else args.oracle
    if args.run_name:
        upd["run_name"] = args.run_name
    if args.variants:
        upd["variants"] = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    if args.datasets:
        upd["datasets"] = tuple(v.strip() for v in args.datasets.split(",") if v.strip())
    if args.n_samples:
        upd["n_samples"] = _int_or_json(args.n_samples, cfg.n_samples)
    if args.max_new_tokens:
        upd["max_new_tokens"] = _int_or_json(args.max_new_tokens, cfg.max_new_tokens)
    if args.device:
        upd["device"] = args.device
    if args.dtype:
        upd["dtype"] = args.dtype
    if args.seed is not None:
        upd["seed"] = args.seed
    if args.results_root:
        upd["results_root"] = args.results_root
    if args.robustness:
        upd["run_robustness"] = True
    if args.chat_template != "auto":
        upd["use_chat_template"] = args.chat_template == "on"
    if args.semstamp_n_max is not None:
        upd["semstamp_n_max"] = args.semstamp_n_max
    return cfg.with_updates(**upd)


def setup_logging(cfg: BenchmarkConfig, stage: str, level: str) -> None:
    log_dir = cfg.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    name = f"{stage.replace(',', '+')}_{job}{'_' + task if task else ''}.log"
    handlers = [logging.StreamHandler(sys.stderr), logging.FileHandler(log_dir / name)]
    logging.basicConfig(level=getattr(logging, level.upper()), handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.print_config:
        print(json.dumps(cfg.as_dict(), indent=2, default=str))
        print("config_hash:", cfg.config_hash())
        return 0
    stages = tuple(STAGES) if args.stage == "all" else tuple(s.strip() for s in args.stage.split(","))
    setup_logging(cfg, args.stage, args.log_level)
    log = logging.getLogger("wmbench.cli")
    log.info("config: %s", json.dumps(cfg.as_dict(), default=str))
    runner = Runner(cfg, resume=not args.no_resume, allow_device_mismatch=args.allow_device_mismatch,
                    force_new_run=args.force_new_run, show_progress=not args.no_progress)
    runner.run(stages)
    log.info("done: %s", cfg.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
