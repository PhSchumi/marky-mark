"""Cached, seeded, timed generation. One JSONL file per (dataset, variant)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from wmbench.config import SCHEMA_VERSION, BenchmarkConfig
from wmbench.data import Sample, TaskSpec, postprocess
from wmbench.models import PromptFormatter
from wmbench.watermarks import WatermarkVariant

log = logging.getLogger("wmbench.generation")

MAX_CONSECUTIVE_ERRORS = 5


# ----------------------------------------------------------------------------- storage


class JsonlStore:
    """Append-only JSONL with fsync per record; a truncated final line is ignored on read."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("dropping malformed line in %s", self.path)
        return rows

    def write(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rewrite(self, records: Iterable[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")
        os.replace(tmp, self.path)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def latest_by_sample(rows: list[dict], *, ok_only: bool = False) -> dict[str, dict]:
    """Last record per sample id (later rows supersede earlier ones, e.g. retried errors)."""
    out: dict[str, dict] = {}
    for r in rows:
        if ok_only and r.get("error"):
            continue
        out[r["sample_id"]] = r
    return out


# ----------------------------------------------------------------------------- seeding


def stable_seed(*parts) -> int:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info(device: str, dtype: torch.dtype) -> dict:
    import transformers

    return {
        "device_type": device.split(":")[0],
        "dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


# ----------------------------------------------------------------------------- generation


def generation_path(run_dir: Path, dataset: str, variant: str) -> Path:
    return Path(run_dir) / "generations" / dataset / f"{variant}.jsonl"


def generate_records(variant: WatermarkVariant, samples: list[Sample], formatter: PromptFormatter,
                     task: TaskSpec, cfg: BenchmarkConfig, env: dict, *, show_progress: bool = True,
                     resume: bool = True) -> list[dict]:
    """Generate (or load cached) records for one (variant, dataset).

    Records with an ``error`` are retried on resume. Aborts after
    :data:`MAX_CONSECUTIVE_ERRORS` consecutive failures or on any CUDA runtime error
    other than an out-of-memory that a single retry resolves.
    """
    store = JsonlStore(generation_path(cfg.run_dir, task.name, variant.name))
    existing = latest_by_sample(store.read(), ok_only=True) if resume else {}
    todo = [s for s in samples if s.id not in existing]
    log.info("%s/%s: %d cached, %d to generate", task.name, variant.name, len(existing), len(todo))
    variant_config_hash = hashlib.sha1(json.dumps(variant.effective_config(), sort_keys=True,
                                                  default=str).encode()).hexdigest()[:12]

    if todo:  # untimed warm-up so lazy initialisation does not land in the first timing
        try:
            variant.generate(formatter.encode(todo[0].task_prompt), formatter.format(todo[0].task_prompt))
        except Exception as exc:  # noqa: BLE001
            log.warning("warm-up generation failed for %s: %s", variant.name, exc)

    consecutive_errors = 0
    iterator = tqdm(todo, desc=f"{task.name}/{variant.name}", disable=not show_progress or not _is_tty())
    for sample in iterator:
        prompt_text = formatter.format(sample.task_prompt)
        prompt_ids = formatter.encode(sample.task_prompt)
        rec = {
            "schema_version": SCHEMA_VERSION, "run_name": cfg.run_name, "dataset": task.name,
            "variant": variant.name, "variant_config_hash": variant_config_hash,
            "sample_id": sample.id, "index": samples.index(sample),
            "seed": stable_seed(cfg.seed, task.name, sample.id, variant.name),
            "prompt": sample.task_prompt, "prompt_text": prompt_text,
            "prompt_sha1": hashlib.sha1(prompt_text.encode()).hexdigest()[:12],
            "prompt_tokens": len(prompt_ids), "prompt_truncated": bool(sample.meta.get("prompt_truncated", False)),
            "reference": sample.reference, "human_text": sample.human_text,
            "continuation": None, "answer": None, "continuation_tokens": None,
            "gen_seconds": None, "tokens_per_second": None, "error": None, **env,
        }
        try:
            seed_everything(rec["seed"])
            t0 = time.perf_counter()
            try:
                out = variant.generate(prompt_ids, prompt_text)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("CUDA OOM on %s; retrying once", sample.id)
                seed_everything(rec["seed"])
                t0 = time.perf_counter()
                out = variant.generate(prompt_ids, prompt_text)
            dt = time.perf_counter() - t0
            rec.update(
                continuation=out.continuation, answer=postprocess(task.kind, out.continuation),
                continuation_tokens=len(out.continuation_ids), gen_seconds=dt,
                tokens_per_second=(len(out.continuation_ids) / dt) if dt > 0 else None,
            )
            consecutive_errors = 0
        except RuntimeError as exc:
            if "CUDA" in str(exc) or "cuda" in str(exc):
                raise  # a poisoned CUDA context cannot be recovered from inside the loop
            rec["error"] = f"{type(exc).__name__}: {exc}"
            consecutive_errors += 1
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            consecutive_errors += 1
        if rec["error"]:
            log.error("%s/%s sample %s failed: %s", task.name, variant.name, sample.id, rec["error"])
        store.write(rec)
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            raise RuntimeError(f"{consecutive_errors} consecutive generation failures for {variant.name}; aborting")

    rows = latest_by_sample(store.read())
    ordered = [rows[s.id] for s in samples if s.id in rows]
    n_ok = sum(1 for r in ordered if not r.get("error"))
    if ordered and n_ok == 0:
        raise RuntimeError(f"no successful generations for {task.name}/{variant.name}")
    return ordered


def _is_tty() -> bool:
    try:
        return os.isatty(2) or "ipykernel" in str(type(get_ipython()))  # type: ignore[name-defined]
    except Exception:  # noqa: BLE001
        return os.isatty(2)
