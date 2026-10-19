"""Benchmark stages. Everything lives under ``results/<run_name>/`` and every stage resumes.

Layout::

    results/<run>/manifest.json                     resolved config, hashes, environment
    results/<run>/generations/<dataset>/<variant>.jsonl
    results/<run>/attacks/<dataset>/<attack>/<variant>.jsonl   attacked texts + strength metrics
    results/<run>/detection/<dataset>/<variant>.jsonl   one row per (sample_id, text_kind)
    results/<run>/quality/<dataset>/<variant>.jsonl
    results/<run>/summary.csv | summary.md | results.json | plots/*.png | logs/

Stages:

* ``generate``     per (dataset, variant): generate and cache. Independent across
                   (dataset, variant) pairs -> SLURM array friendly.
* ``attack``       (``run_robustness``) per attack and watermark variant: attack the common-budget
                   watermarked text of every sample, record strength metrics; cached under
                   ``attacks/<dataset>/<attack>/<variant>.jsonl``. Loads one attacker at a time.
* ``detect``       per watermark variant on the detection datasets: score the own, the human
                   and the ``none`` continuations of every sample at one common token budget
                   (plus the ``attacked:<name>`` texts from stage ``attack``). Needs the ``none``
                   generations and the variant's generations, so it runs after the ``generate`` array.
* ``quality``      PPL (oracle), diversity, BLEU, ROUGE, pass@1, GSM8K accuracy for every variant (+ human row).
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
from wmbench.attacks import ATTACKS, attack_config_digest, build_attacker, load_similarity_scorer
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

STAGES = ("generate", "attack", "detect", "quality", "report")
TEXT_KINDS = ("watermarked", "human", "unwatermarked")  # plus "attacked:<attack>" per configured attack

try:  # POSIX advisory lock for the shared manifest (SLURM array tasks); no-op elsewhere
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


def detection_path(run_dir: Path, dataset: str, variant: str) -> Path:
    return Path(run_dir) / "detection" / dataset / f"{variant}.jsonl"


def quality_path(run_dir: Path, dataset: str, variant: str) -> Path:
    return Path(run_dir) / "quality" / dataset / f"{variant}.jsonl"


def attack_path(run_dir: Path, dataset: str, attack: str, variant: str) -> Path:
    return Path(run_dir) / "attacks" / dataset / attack / f"{variant}.jsonl"


def truncate_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def common_token_budget(tokenizer, texts: dict[str, str | None], budget: int) -> tuple[int, dict[str, str]]:
    """Cut every text to the token count of the shortest one, capped at ``budget``.

    Detection scores accumulate evidence per token, so the watermarked, human and unwatermarked
    texts of a sample must be scored at one common length ``T = min(budget, |text|, ...)``;
    otherwise the longer class simply gets more chances to look watermarked. Lengths are measured
    the way the detectors see them (re-tokenised without special tokens); ``None`` counts as
    empty. Texts already at or below ``T`` are returned unchanged.
    """
    ids = {k: tokenizer(t or "", add_special_tokens=False)["input_ids"] for k, t in texts.items()}
    t = min([int(budget), *(len(v) for v in ids.values())])
    out = {k: (texts[k] or "") if len(v) <= t else tokenizer.decode(v[:t], skip_special_tokens=True)
           for k, v in ids.items()}
    return t, out


def attack_records(attacker, tokenizer, dataset: str, variant: str, records: list[dict], none_by_id: dict[str, dict],
                   cfg: BenchmarkConfig, env: dict, store: JsonlStore, *, similarity=None, resume: bool = True,
                   show_progress: bool = True) -> dict[str, int]:
    """Attack the common-budget watermarked text of every sample of one variant and cache the result.

    The attacked text is derived from exactly the text that stage ``detect`` scores clean (the
    watermarked continuation cut to the sample's common budget ``T``); the raw task prompt is the
    attacker's context. The output is capped at ``T`` tokens so a longer paraphrase cannot add
    detection evidence (``n_tokens_uncapped`` keeps the original length). Per row: ``len_ratio``
    (uncapped tokens / T), ``bleu_vs_source`` (sacreBLEU, model-free) and ``sim_embed`` (cosine
    similarity from ``similarity(source, attacked)`` when a scorer is given). Seed:
    ``stable_seed(seed, "attack", attack, dataset, sample_id)``, identical across variants.
    Resume: a row is done when it has no error, the current ``attack_config_hash`` and the same
    ``T``; error rows are retried.
    """
    ahash = attack_config_digest(attacker)
    budget = cfg.max_new_tokens[dataset]
    done: dict[str, int] = {}
    if resume:
        for r in store.read():
            if not r.get("error") and r.get("attack_config_hash") == ahash and r.get("detect_tokens") is not None:
                done[r["sample_id"]] = int(r["detect_tokens"])
    todo: list[tuple[dict, int, str]] = []
    skipped_no_none = skipped_empty = cached = errors = 0
    for rec in records:
        if rec.get("error"):
            continue
        none_rec = none_by_id.get(rec["sample_id"])
        if none_rec is None or none_rec.get("error"):
            skipped_no_none += 1
            continue
        t, texts = common_token_budget(tokenizer, {"watermarked": rec["continuation"], "human": rec["human_text"],
                                                   "unwatermarked": none_rec["continuation"]}, budget)
        if t < 1:
            skipped_empty += 1
        elif done.get(rec["sample_id"]) == t:
            cached += 1
        else:
            todo.append((rec, t, texts["watermarked"]))
    log.info("attack %s on %s/%s: %d texts to attack, %d cached; skipped %d samples without a 'none' generation "
             "and %d with an empty common budget", attacker.name, dataset, variant, len(todo), cached,
             skipped_no_none, skipped_empty)
    for rec, t, source in tqdm(todo, desc=f"attack {attacker.name} {dataset}/{variant}", disable=not show_progress):
        seed = stable_seed(cfg.seed, "attack", attacker.name, dataset, rec["sample_id"])
        row = {"schema_version": SCHEMA_VERSION, "dataset": dataset, "variant": variant, "attack": attacker.name,
               "attack_config_hash": ahash, "sample_id": rec["sample_id"], "detect_tokens": t, "seed": seed,
               "text": None, "n_tokens": None, "n_tokens_uncapped": None, "len_ratio": float("nan"),
               "bleu_vs_source": float("nan"), "sim_embed": None, "seconds": None, "error": None,
               "device_type": env["device_type"], "dtype": env["dtype"]}
        try:
            t0 = time.perf_counter()
            attacked = attacker.attack(source, rec.get("prompt") or "", seed)
            row["seconds"] = time.perf_counter() - t0
            n_uncapped = len(tokenizer(attacked, add_special_tokens=False)["input_ids"])
            attacked = truncate_tokens(tokenizer, attacked, t)
            row.update(text=attacked, n_tokens=len(tokenizer(attacked, add_special_tokens=False)["input_ids"]),
                       n_tokens_uncapped=n_uncapped, len_ratio=n_uncapped / t,
                       bleu_vs_source=metrics.sentence_bleu(attacked, source),
                       sim_embed=float(similarity(source, attacked)) if similarity is not None and attacked else None)
        except Exception as exc:  # noqa: BLE001 - one failed paraphrase must not abort the track
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors += 1
            log.error("attack %s %s/%s sample %s failed: %s", attacker.name, dataset, variant, rec["sample_id"], row["error"])
        store.write(row)
    return {"attacked": len(todo), "cached": cached, "skipped_no_none": skipped_no_none,
            "skipped_empty": skipped_empty, "errors": errors}


def detect_records(variant, tokenizer, dataset: str, records: list[dict], none_by_id: dict[str, dict],
                   cfg: BenchmarkConfig, env: dict, store: JsonlStore, *, attacks: dict[str, dict[str, dict]] | None = None,
                   resume: bool = True, show_progress: bool = True) -> dict[str, int]:
    """Score one watermark variant's samples at a common per-sample token budget.

    For every sample with a usable watermarked record and a usable ``none`` record, the
    watermarked continuation, the human continuation and the ``none`` continuation are cut to
    ``T = min(max_new_tokens, len(each))`` (:func:`common_token_budget`) and scored at that
    length. ``attacks`` maps an attack name to its cached rows (``{sample_id: row}`` from stage
    ``attack``); a row is scored as ``attacked:<name>`` only when it has no error and was derived
    at the same ``T`` (stale rows are counted and skipped). Every row records ``T`` as
    ``detect_tokens`` (``n_tokens`` is the count the detector actually saw after the decode/encode
    round trip) and, for attacked kinds, the ``attack_config_hash``. Resume: a row is done when it
    has no error, the variant's current hyperparameters, the same ``T`` and (attacked kinds) the
    same attack hash; rows without ``detect_tokens`` predate length control and are re-scored.
    Samples without a ``none`` record or with an empty common budget are skipped and counted.
    """
    vhash = variant_config_digest(variant)
    budget = cfg.max_new_tokens[dataset]
    attacks = attacks or {}
    done: dict[tuple[str, str], tuple[int, str | None]] = {}
    if resume:
        for r in store.read():
            if not r.get("error") and r.get("variant_config_hash") == vhash and r.get("detect_tokens") is not None:
                done[(r["sample_id"], r["text_kind"])] = (int(r["detect_tokens"]), r.get("attack_config_hash"))
    todo: list[tuple[str, str, int, str, str | None]] = []
    skipped_no_none = skipped_empty = skipped_stale_attack = skipped_missing_attack = cached = 0
    for rec in records:
        if rec.get("error"):
            continue
        sid = rec["sample_id"]
        none_rec = none_by_id.get(sid)
        if none_rec is None or none_rec.get("error"):
            skipped_no_none += 1
            continue
        t, texts = common_token_budget(tokenizer, {"watermarked": rec["continuation"], "human": rec["human_text"],
                                                   "unwatermarked": none_rec["continuation"]}, budget)
        if t < 1:
            skipped_empty += 1
            continue
        items: list[tuple[str, str, str | None]] = [(kind, texts[kind], None) for kind in TEXT_KINDS]
        for name, rows in attacks.items():
            row = rows.get(sid)
            if row is None:
                skipped_missing_attack += 1
            elif row.get("error") or row.get("detect_tokens") != t or not row.get("text"):
                skipped_stale_attack += 1
            else:
                items.append((f"attacked:{name}", row["text"], row.get("attack_config_hash")))
        for kind, text, ahash in items:
            if done.get((sid, kind)) == (t, ahash):
                cached += 1
            else:
                todo.append((sid, kind, t, text, ahash))
    log.info("detect %s/%s: %d texts to score, %d cached; skipped %d samples without a 'none' generation, %d with an "
             "empty common budget, %d stale and %d missing attack rows", dataset, variant.name, len(todo), cached,
             skipped_no_none, skipped_empty, skipped_stale_attack, skipped_missing_attack)
    for sid, kind, t, text, ahash in tqdm(todo, desc=f"detect {dataset}/{variant.name}", disable=not show_progress):
        d = variant.detect(text)
        store.write({
            "schema_version": SCHEMA_VERSION, "dataset": dataset, "variant": variant.name,
            "variant_config_hash": vhash, "sample_id": sid, "text_kind": kind, "detect_tokens": t,
            "attack_config_hash": ahash, "score": d.score, "is_watermarked": d.is_watermarked,
            "n_tokens": d.n_tokens, "error": d.error, "raw": d.raw,
            "device_type": env["device_type"], "dtype": env["dtype"],
        })
    return {"scored": len(todo), "cached": cached, "skipped_no_none": skipped_no_none, "skipped_empty": skipped_empty,
            "skipped_stale_attack": skipped_stale_attack, "skipped_missing_attack": skipped_missing_attack}


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
                generate_records(v, samples, self.formatter, task, self.cfg, self.env,
                                 show_progress=self.show_progress, resume=self.resume)
                del v
                gc.collect()
        self.write_manifest(stage="generate")

    # ------------------------------------------------------------------ cached generation records
    def _none_records(self, dataset: str) -> dict[str, dict]:
        rows = JsonlStore(generation_path(self.run_dir, dataset, "none")).read()
        if not rows:
            raise FileNotFoundError(f"no 'none' generations for {dataset}; run stage 'generate' with variant none first")
        none_by_id = latest_by_sample(rows)
        self._check_device(list(none_by_id.values()))
        return none_by_id

    def _variant_records(self, dataset: str, name: str) -> list[dict]:
        rows = JsonlStore(generation_path(self.run_dir, dataset, name)).read()
        if not rows:
            raise FileNotFoundError(f"no generations for {dataset}/{name}; run stage 'generate' first")
        recs = list(latest_by_sample(rows).values())
        self._check_device(recs)
        return recs

    def _attack_rows(self, dataset: str, name: str) -> dict[str, dict[str, dict]]:
        """Latest attack row per sample for every configured attack; a missing file is a warning."""
        out: dict[str, dict[str, dict]] = {}
        for attack in self.cfg.attacks:
            rows = JsonlStore(attack_path(self.run_dir, dataset, attack, name)).read()
            if not rows:
                log.warning("no '%s' attack rows for %s/%s; run stage 'attack' first", attack, dataset, name)
                continue
            out[attack] = latest_by_sample(rows)
        return out

    # ------------------------------------------------------------------ stage: attack
    def stage_attack(self, variants=None) -> None:
        """Attack the watermarked texts with every configured attack (see :func:`attack_records`)."""
        if not self.cfg.run_robustness:
            log.info("stage attack: run_robustness is off, nothing to do")
            return
        for dataset in DETECTION_DATASETS:
            if dataset not in self.cfg.datasets:
                continue
            none_by_id = self._none_records(dataset)
            names = [n for n in self.variant_names_for(dataset, variants) if n != "none"]
            gens = {name: self._variant_records(dataset, name) for name in names}
            similarity = (load_similarity_scorer(self.cfg.attack_similarity_model, self.device)
                          if self.cfg.attack_similarity_model else None)
            for attack in self.cfg.attacks:
                if ATTACKS[attack].needs_model:
                    self.free_generator()  # the paraphraser and the generator rarely fit together
                attacker = build_attacker(attack, self.cfg, self.device)
                try:
                    for name in names:
                        attack_records(attacker, self.tokenizer, dataset, name, gens[name], none_by_id, self.cfg,
                                       self.env, JsonlStore(attack_path(self.run_dir, dataset, attack, name)),
                                       similarity=similarity, resume=self.resume, show_progress=self.show_progress)
                finally:
                    attacker.close()
                    gc.collect()
        self.write_manifest(stage="attack")

    # ------------------------------------------------------------------ stage: detect
    def stage_detect(self, variants=None) -> None:
        """Length-controlled detection for every watermark variant (see :func:`detect_records`)."""
        for dataset in DETECTION_DATASETS:
            if dataset not in self.cfg.datasets:
                continue
            none_by_id = self._none_records(dataset)
            for name in self.variant_names_for(dataset, variants):
                if name == "none":
                    continue
                recs = self._variant_records(dataset, name)
                attacks = self._attack_rows(dataset, name) if self.cfg.run_robustness else None
                v = self.build(name, dataset)
                detect_records(v, self.tokenizer, dataset, recs, none_by_id, self.cfg, self.env,
                               JsonlStore(detection_path(self.run_dir, dataset, name)), attacks=attacks,
                               resume=self.resume, show_progress=self.show_progress)
                del v
                gc.collect()
        self.write_manifest(stage="detect")

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
                elif task.kind == "math":
                    out.update(correct=metrics.math_answer_correct(rec["answer"], rec["reference"]),
                               answer=rec["answer"], gold=rec["reference"])
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
    def run(self, stages=STAGES, variants=None, datasets=None):
        for stage in stages:
            if stage == "detect_none":  # pre-schema-2 name
                log.warning("stage 'detect_none' is now 'detect': it scores the watermarked, human and none texts "
                            "of every sample at one common token budget")
                stage = "detect"
            log.info("=== stage %s ===", stage)
            if stage == "generate":
                self.stage_generate(variants, datasets)
            elif stage == "attack":
                self.stage_attack(variants)
            elif stage == "detect":
                self.stage_detect(variants)
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
