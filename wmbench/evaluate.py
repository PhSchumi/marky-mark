"""Benchmark stages. Everything lives under ``results/<run_name>/`` and every stage resumes.

Layout::

    results/<run>/manifest.json                     resolved config, hashes, environment
    results/<run>/generations/<dataset>/<variant>.jsonl
    results/<run>/detection/<dataset>/<variant>.jsonl   one row per (sample_id, text_kind)
    results/<run>/quality/<dataset>/<variant>.jsonl
    results/<run>/summary.csv | summary.md | results.json | plots/*.png | logs/

Stages:

* ``generate``     per (dataset, variant): generate; for watermark variants on detection
                   datasets also score the own continuations, the human continuations and
                   (optionally) word-deletion-attacked continuations. Independent across
                   (dataset, variant) pairs -> SLURM array friendly.
* ``detect_none``  each watermark variant scores the ``none`` continuations (needs ``none``).
* ``quality``      PPL (oracle), diversity, BLEU, ROUGE, pass@1 for every variant (+ human row).
* ``report``       pure pandas over the JSONL files.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import tempfile
import time
import weakref
from pathlib import Path

import torch
from tqdm.auto import tqdm

from wmbench import __version__, metrics
from wmbench.attacks import word_deletion
from wmbench.bootstrap import MARKLLM_COMMIT, REPO_ROOT, ensure_markllm
from wmbench.config import DETECTION_DATASETS, SCHEMA_VERSION, BenchmarkConfig
from wmbench.data import TASKS, PromptTemplates, Sample, load_task
from wmbench.generation import (
    JsonlStore,
    environment_info,
    generate_records,
    generation_path,
    latest_by_sample,
    stable_seed,
    variant_config_digest,
)
from wmbench.models import PromptFormatter, load_causal_lm, load_tokenizer, pick_device, pick_dtype
from wmbench.watermarks import VARIANTS, WatermarkVariant, build_variant

log = logging.getLogger("wmbench.evaluate")

STAGES = ("generate", "detect_none", "quality", "report")
TEXT_KINDS = ("watermarked", "human", "unwatermarked", "attacked")

try:  # POSIX advisory lock for the shared manifest (SLURM array tasks); no-op elsewhere
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


def detection_path(run_dir: Path, dataset: str, variant: str) -> Path:
    return Path(run_dir) / "detection" / dataset / f"{variant}.jsonl"


def quality_path(run_dir: Path, dataset: str, variant: str) -> Path:
    return Path(run_dir) / "quality" / dataset / f"{variant}.jsonl"


def truncate_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _repo_git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


class _ManifestLock:
    """Exclusive advisory lock on ``<run>/.manifest.lock`` (best effort on NFS)."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / ".manifest.lock"
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        if fcntl is not None:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if fcntl is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


class Runner:
    """Executes benchmark stages for one :class:`BenchmarkConfig`."""

    def __init__(self, cfg: BenchmarkConfig, *, resume: bool = True, allow_device_mismatch: bool = False,
                 force_new_run: bool = False, show_progress: bool = True):
        self.cfg = cfg
        self.resume = resume
        self.allow_device_mismatch = allow_device_mismatch
        self.show_progress = show_progress
        self.device = pick_device(cfg.device)
        self.dtype = pick_dtype(self.device, cfg.dtype)
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._model = self._tok = None
        self._oracle = None
        self._samples: dict[str, list[Sample]] = {}
        ensure_markllm()
        self.env = environment_info(self.device, self.dtype)
        if cfg.use_chat_template is None and not self.formatter.is_chat:
            log.info("no chat template found for %s: using base-model prompts (few-shot / cue words)", cfg.model_name)
        self.write_manifest(force_new_run=force_new_run)

    # ------------------------------------------------------------------ manifest
    def write_manifest(self, *, force_new_run: bool = False, stage: str | None = None) -> dict:
        """Create or update ``manifest.json`` atomically under a lock (array tasks share it)."""
        path = self.run_dir / "manifest.json"
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        manifest = {
            "schema_version": SCHEMA_VERSION, "run_name": self.cfg.run_name, "config": self.cfg.as_dict(),
            "config_hash": self.cfg.config_hash(), "markllm_commit": MARKLLM_COMMIT,
            "wmbench_version": __version__, "repo_git_sha": _repo_git_sha(), "environment": self.env,
            "resolved": {"device": self.device, "dtype": self.env["dtype"], "chat_template": self.formatter.is_chat,
                         "oracle_model_name": self.cfg.oracle_model_name or self.cfg.model_name},
            "created": now, "updated": now, "stages_run": [],
        }
        with _ManifestLock(self.run_dir):
            old = None
            if path.exists():
                try:
                    old = json.loads(path.read_text()) or None
                except json.JSONDecodeError:
                    log.warning("manifest.json is corrupt; rewriting it")
            if old:
                if old.get("config_hash") != manifest["config_hash"] and not force_new_run:
                    raise RuntimeError(
                        f"results dir {self.run_dir} was produced with config hash {old.get('config_hash')} "
                        f"but the current config hashes to {manifest['config_hash']} (model, seed, lengths, prompt "
                        f"settings or configs/watermarks/*.json changed). Use a new run_name or pass "
                        f"force_new_run/--force-new-run to append anyway.")
                manifest["created"] = old.get("created", now)
                manifest["stages_run"] = old.get("stages_run", [])
                old_env = old.get("environment", {})
                for key, label in (("device_type", "device type"), ("dtype", "dtype")):
                    if old_env.get(key) != self.env[key] and not self.allow_device_mismatch:
                        raise RuntimeError(
                            f"run {self.cfg.run_name} was generated with {label} {old_env.get(key)} but this process "
                            f"resolves to {self.env[key]}. Watermark RNG streams and logits differ, so all stages "
                            f"must use the same {label}. Pass allow_device_mismatch=True to override.")
            if stage:
                manifest["stages_run"].append({"stage": stage, "time": now, "slurm": self.env.get("slurm_job_id"),
                                               "variants": list(self.cfg.variants), "datasets": list(self.cfg.datasets)})
            fd, tmp = tempfile.mkstemp(dir=self.run_dir, prefix="manifest.", suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(manifest, indent=2, default=str))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        return manifest

    # ------------------------------------------------------------------ models
    @property
    def model(self):
        if self._model is None:
            self._model, self._tok = load_causal_lm(
                self.cfg.model_name, self.device, self.dtype, trust_remote_code=self.cfg.trust_remote_code,
                tokenizer_name=self.cfg.tokenizer_name, tokenizer=self._tok)
        return self._model

    @property
    def tokenizer(self):
        """The generator's tokenizer, loaded without the model (quality-only jobs never need the LM)."""
        if self._tok is None:
            self._tok = load_tokenizer(self.cfg.tokenizer_name or self.cfg.model_name,
                                       trust_remote_code=self.cfg.trust_remote_code)
        return self._tok

    def free_generator(self) -> None:
        """Drop the generator; warn if something else (e.g. notebook variables) still references it."""
        ref = weakref.ref(self._model) if self._model is not None else None
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if ref is not None and ref() is not None:
            log.warning("the generator model is still referenced elsewhere (notebook variables?) and was NOT "
                        "freed before loading the oracle; `del` those references to release GPU memory")

    @property
    def formatter(self) -> PromptFormatter:
        return PromptFormatter(self.tokenizer, self.cfg.use_chat_template, self.cfg.system_prompt)

    @property
    def templates(self) -> PromptTemplates:
        return PromptTemplates(chat=self.formatter.is_chat, translation_shots=self.cfg.translation_shots,
                               article_max_words=self.cfg.article_max_words)

    def samples(self, dataset: str) -> list[Sample]:
        if dataset not in self._samples:
            self._samples[dataset] = load_task(dataset, self.cfg.n_samples[dataset], self.templates,
                                               shuffle_seed=self.cfg.shuffle_seed)
        return self._samples[dataset]

    def variant_names_for(self, dataset: str, variants=None) -> list[str]:
        names = list(variants or self.cfg.variants)
        return [v for v in names if VARIANTS[v].datasets is None or dataset in VARIANTS[v].datasets]

    def build(self, name: str, dataset: str) -> WatermarkVariant:
        return build_variant(name, self.model, self.tokenizer, self.device, self.cfg.max_new_tokens[dataset],
                             overrides=self.cfg.watermark_overrides.get(name), semstamp_n_max=self.cfg.semstamp_n_max)

    # ------------------------------------------------------------------ stage: generate
    def stage_generate(self, variants=None, datasets=None) -> None:
        for dataset in datasets or self.cfg.datasets:
            task = TASKS[dataset]
            samples = self.samples(dataset)
            for name in self.variant_names_for(dataset, variants):
                v = self.build(name, dataset)
                recs = generate_records(v, samples, self.formatter, task, self.cfg, self.env,
                                        show_progress=self.show_progress, resume=self.resume)
                if v.is_watermarked and dataset in DETECTION_DATASETS:
                    kinds = ["watermarked", "human"] + (["attacked"] if self.cfg.run_robustness else [])
                    self._detect(v, dataset, recs, kinds)
                del v
                gc.collect()
        self.write_manifest(stage="generate")

    def _detect(self, v: WatermarkVariant, dataset: str, records: list[dict], kinds: list[str]) -> None:
        store = JsonlStore(detection_path(self.run_dir, dataset, v.name))
        vhash = variant_config_digest(v)
        done = set()
        if self.resume:
            done = {(r["sample_id"], r["text_kind"]) for r in store.read()
                    if not r.get("error") and r.get("variant_config_hash") == vhash}
        budget = self.cfg.max_new_tokens[dataset]
        todo = [(r, k) for r in records if not r.get("error") for k in kinds if (r["sample_id"], k) not in done]
        log.info("detect %s/%s: %d texts to score", dataset, v.name, len(todo))
        for rec, kind in tqdm(todo, desc=f"detect {dataset}/{v.name}", disable=not self.show_progress):
            if kind in ("watermarked", "unwatermarked"):
                text = rec["continuation"]
            elif kind == "human":
                text = truncate_tokens(self.tokenizer, rec["human_text"] or "", budget)
            elif kind == "attacked":
                text = word_deletion(rec["continuation"], self.cfg.word_deletion_ratio,
                                     stable_seed(self.cfg.seed, "attack", dataset, rec["sample_id"]))
            else:
                raise ValueError(kind)
            d = v.detect(text)
            store.write({
                "schema_version": SCHEMA_VERSION, "dataset": dataset, "variant": v.name, "variant_config_hash": vhash,
                "sample_id": rec["sample_id"], "text_kind": kind, "score": d.score,
                "is_watermarked": d.is_watermarked, "n_tokens": d.n_tokens, "error": d.error, "raw": d.raw,
                "device_type": self.env["device_type"], "dtype": self.env["dtype"],
            })

    # ------------------------------------------------------------------ stage: detect_none
    def stage_detect_none(self, variants=None) -> None:
        for dataset in DETECTION_DATASETS:
            if dataset not in self.cfg.datasets:
                continue
            none_rows = JsonlStore(generation_path(self.run_dir, dataset, "none")).read()
            if not none_rows:
                raise FileNotFoundError(f"no 'none' generations for {dataset}; run stage 'generate' with variant none first")
            none_recs = [r for r in latest_by_sample(none_rows).values()]
            self._check_device(none_recs)
            for name in self.variant_names_for(dataset, variants):
                if name == "none":
                    continue
                v = self.build(name, dataset)
                self._detect(v, dataset, none_recs, ["unwatermarked"])
                del v
                gc.collect()
        self.write_manifest(stage="detect_none")

    def _check_device(self, records: list[dict]) -> None:
        devs = {r.get("device_type") for r in records if r.get("device_type")}
        if devs and devs != {self.env["device_type"]} and not self.allow_device_mismatch:
            raise RuntimeError(f"records were generated on {devs}, detection runs on {self.env['device_type']}")

    # ------------------------------------------------------------------ stage: quality
    @property
    def oracle_name(self) -> str:
        return self.cfg.oracle_model_name or self.cfg.model_name

    def _oracle_model(self):
        if self.cfg.oracle_model_name in (None, "", self.cfg.model_name):
            return self.model, self.tokenizer, self.cfg.model_name
        if self._oracle is None:
            self.free_generator()  # generator and oracle rarely fit together; generation is done by now
            m, t = load_causal_lm(self.cfg.oracle_model_name, self.device, self.dtype,
                                  trust_remote_code=self.cfg.trust_remote_code)
            self._oracle = (m, t, self.cfg.oracle_model_name)
        return self._oracle

    def stage_quality(self, variants=None, datasets=None) -> None:
        for dataset in datasets or self.cfg.datasets:
            task = TASKS[dataset]
            names = self.variant_names_for(dataset, variants)
            if task.kind == "completion":
                names = names + ["human"]
            for name in names:
                self._quality_one(dataset, task, name)
        self.write_manifest(stage="quality")

    def _quality_one(self, dataset: str, task, name: str) -> None:
        src_variant = "none" if name == "human" else name
        rows = JsonlStore(generation_path(self.run_dir, dataset, src_variant)).read()
        if not rows:
            log.warning("no generations for %s/%s; skipping quality", dataset, src_variant)
            return
        recs = [r for r in latest_by_sample(rows).values() if not r.get("error")]
        store = JsonlStore(quality_path(self.run_dir, dataset, name))
        done = set()
        if self.resume:
            done = {r["sample_id"] for r in store.read()
                    if not r.get("error") and (task.kind != "completion" or r.get("oracle_model") == self.oracle_name)}
        todo = [r for r in recs if r["sample_id"] not in done]
        if not todo:
            return
        if task.kind == "code":
            self._check_humaneval_runner(todo[0])
        oracle = self._oracle_model() if "ppl" in task.quality_metrics else None
        prompt_special = not self.formatter.is_chat  # chat prompt text already carries its special tokens
        budget = self.cfg.max_new_tokens[dataset]
        for rec in tqdm(todo, desc=f"quality {dataset}/{name}", disable=not self.show_progress):
            out = {"schema_version": SCHEMA_VERSION, "dataset": dataset, "variant": name,
                   "sample_id": rec["sample_id"], "error": None}
            try:
                if task.kind == "completion":
                    text = rec["continuation"] if name != "human" else truncate_tokens(
                        self.tokenizer, rec["human_text"] or "", budget)
                    m, t, oname = oracle
                    nll, n = metrics.continuation_nll(m, t, rec["prompt_text"], text, self.device,
                                                      prompt_add_special_tokens=prompt_special)
                    out.update(oracle_model=oname, nll_sum=nll, n_tokens=n,
                               ppl=(metrics.math.exp(nll / n) if n else float("nan")), **metrics.diversity_metrics(text))
                elif task.kind == "translation":
                    out.update(bleu_sentence=metrics.sentence_bleu(rec["answer"], rec["reference"]),
                               answer_words=len(rec["answer"].split()))
                elif task.kind == "summarization":
                    out.update(**metrics.rouge_scores(rec["answer"], rec["reference"]),
                               answer_words=len(rec["answer"].split()))
                elif task.kind == "code":
                    ref = rec["reference"]
                    out.update(passed=metrics.humaneval_passes(ref["task"], rec["answer"], ref["test"], ref["entry_point"]))
            except Exception as exc:  # noqa: BLE001
                out["error"] = f"{type(exc).__name__}: {exc}"
            store.write(out)

    def _check_humaneval_runner(self, rec: dict) -> None:
        """The canonical solution of one task must pass, otherwise the subprocess environment is broken."""
        ref, canonical = rec["reference"], rec.get("human_text")
        if not canonical:
            return
        if not metrics.humaneval_passes(ref["task"], canonical, ref["test"], ref["entry_point"]):
            raise RuntimeError("HumanEval sanity check failed: the canonical solution of "
                               f"{rec['sample_id']} does not pass in the subprocess runner")

    # ------------------------------------------------------------------ stage: report
    def stage_report(self):
        from wmbench.report import build_report

        out = build_report(self.cfg)
        self.write_manifest(stage="report")
        return out

    # ------------------------------------------------------------------ driver
    def run(self, stages=("generate", "detect_none", "quality", "report"), variants=None, datasets=None):
        for stage in stages:
            log.info("=== stage %s ===", stage)
            if stage == "generate":
                self.stage_generate(variants, datasets)
            elif stage == "detect_none":
                self.stage_detect_none(variants)
            elif stage == "quality":
                self.stage_quality(variants, datasets)
            elif stage == "report":
                self.stage_report()
            else:
                raise ValueError(f"unknown stage {stage}; choose from {STAGES}")


def run_benchmark(cfg: BenchmarkConfig, stages=STAGES, **runner_kw):
    """Convenience wrapper: run all stages sequentially in one process (notebook / single job)."""
    runner = Runner(cfg, **runner_kw)
    runner.run(stages)
    return runner
