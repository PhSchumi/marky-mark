"""Benchmark configuration, presets and environment overrides."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from wmbench.bootstrap import REPO_ROOT

# 2: detection rows carry `detect_tokens`, the common per-sample token budget at which the watermarked,
#    human and `none` texts were all scored (stage `detect`); rows without it are refused by the report.
SCHEMA_VERSION = 2

CONFIG_DIR = REPO_ROOT / "configs" / "watermarks"

# Sampling is fixed to plain multinomial sampling at temperature 1. Only this setting gives
# KGW/SIR (logit bias before tempering), MorphMark/SynthID (return log-probs of a reweighted
# distribution) and PF (own decoding loop) the same base distribution. See README "Fairness".
# repetition_penalty / no_repeat_ngram_size are pinned to their neutral values as documentation;
# the hub generation_config of the model is additionally discarded in models.load_causal_lm.
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}

DEFAULT_VARIANTS = ("none", "KGW", "MorphMark", "PF", "SynthID", "SIR")
DEFAULT_DATASETS = ("c4", "wmt16_de_en", "cnn_dailymail")
DETECTION_DATASETS = ("c4",)  # detection is measured on open-ended C4 continuations only


def watermark_config_digest(config_dir: Path | None = None) -> dict[str, str]:
    """Canonical-JSON digest of every algorithm config file (whitespace edits do not count)."""
    out = {}
    for p in sorted((config_dir or CONFIG_DIR).glob("*.json")):
        try:
            canonical = json.dumps(json.loads(p.read_text()), sort_keys=True)
        except json.JSONDecodeError:
            canonical = p.read_text()
        out[p.name] = hashlib.sha1(canonical.encode()).hexdigest()[:12]
    return out


@dataclass
class BenchmarkConfig:
    run_name: str = "full"
    # --- model ------------------------------------------------------------------------
    model_name: str = "facebook/opt-1.3b"  # PLACEHOLDER: swap for your Hugging Face model
    tokenizer_name: str | None = None
    trust_remote_code: bool = False
    oracle_model_name: str | None = "facebook/opt-2.7b"  # PPL oracle; None = score with the generator
    device: str = "auto"
    dtype: str = "auto"  # auto -> fp16 on CUDA, fp32 elsewhere; or float16/bfloat16/float32
    use_chat_template: bool | None = None  # None = auto-detect from tokenizer
    system_prompt: str | None = None
    # --- what to run -----------------------------------------------------------------
    variants: tuple[str, ...] = DEFAULT_VARIANTS
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    n_samples: dict[str, int] = field(default_factory=lambda: {
        "c4": 500, "wmt16_de_en": 200, "cnn_dailymail": 100, "humaneval": 164})
    max_new_tokens: dict[str, int] = field(default_factory=lambda: {
        "c4": 200, "wmt16_de_en": 64, "cnn_dailymail": 96, "humaneval": 300})
    seed: int = 0
    shuffle_seed: int | None = None  # None = first n rows (MarkLLM convention)
    # --- prompts ----------------------------------------------------------------------
    translation_shots: int = 5
    article_max_words: int = 400
    # --- detection --------------------------------------------------------------------
    min_detect_tokens: int = 50  # report-time: drop samples whose common detection budget (min of the
    # watermarked / human / none lengths, <= max_new_tokens) is shorter; lowering it re-scores on resume
    target_fprs: tuple[float, ...] = (0.01, 0.05)
    # --- optional tracks --------------------------------------------------------------
    run_robustness: bool = False  # stage `attack` + `attacked:<name>` detection rows
    attacks: tuple[str, ...] = ("word_deletion", "dipper")  # keys of wmbench.attacks.ATTACKS
    word_deletion_ratio: float = 0.3
    dipper_model: str = "kalpeshk2011/dipper-paraphraser-xxl"  # T5-XXL, ~22 GB in bf16
    dipper_tokenizer: str = "google/t5-v1_1-xxl"
    dipper_lex_diversity: int = 60  # MarkLLM DipperParaphraser defaults: lex 60, order 0, one sentence per window
    dipper_order_diversity: int = 0
    dipper_sent_interval: int = 1
    dipper_dtype: str = "auto"  # auto -> bf16 on CUDA, fp32 elsewhere (T5 overflows in fp16)
    attack_similarity_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"  # None = no embedding similarity
    semstamp_n_max: int = 100
    # --- per-variant JSON overrides, e.g. {"KGW": {"delta": 1.5}} ---------------------
    watermark_overrides: dict[str, dict] = field(default_factory=dict)
    # --- paths ------------------------------------------------------------------------
    results_root: str = str(REPO_ROOT / "results")
    smoke: bool = False

    # ------------------------------------------------------------------ helpers
    @property
    def run_dir(self) -> Path:
        return Path(self.results_root) / self.run_name

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    # Fields that select *which* work is done, how it is reported, or a device/dtype *preference*
    # (the resolved device type and dtype are checked separately through manifest.json).
    # Array tasks with different --variants / --datasets must share one results directory.
    # Attack settings are excluded too: every attack row carries its own `attack_config_hash`, so a
    # changed paraphraser regenerates exactly the affected rows (word_deletion_ratio stays in the hash
    # so run directories created before the attack track keep their hash).
    _HASH_EXCLUDED = ("run_name", "results_root", "target_fprs", "smoke", "oracle_model_name",
                      "min_detect_tokens", "variants", "datasets", "n_samples", "run_robustness",
                      "device", "dtype", "attacks", "dipper_model", "dipper_tokenizer", "dipper_lex_diversity",
                      "dipper_order_diversity", "dipper_sent_interval", "dipper_dtype", "attack_similarity_model")

    def config_hash(self) -> str:
        """Hash of everything that changes generated records, including the algorithm JSON files."""
        d = self.as_dict()
        for k in self._HASH_EXCLUDED:
            d.pop(k, None)
        d["sampling"] = SAMPLING
        d["watermark_config_files"] = watermark_config_digest()
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def with_updates(self, **kw) -> "BenchmarkConfig":
        return dataclasses.replace(self, **kw)

    # ------------------------------------------------------------------ presets
    @classmethod
    def full(cls, **kw) -> "BenchmarkConfig":
        return cls(**kw)

    @classmethod
    def smoke_test(cls, **kw) -> "BenchmarkConfig":
        """Tiny model, few samples, short generations: validates the pipeline on a CPU in minutes.

        Numbers from this preset are not meaningful; the banner in the report says so.
        """
        base = dict(
            run_name="smoke", model_name="facebook/opt-125m", oracle_model_name=None,
            n_samples={"c4": 8, "wmt16_de_en": 6, "cnn_dailymail": 4, "humaneval": 4},
            max_new_tokens={"c4": 64, "wmt16_de_en": 32, "cnn_dailymail": 48, "humaneval": 128},
            min_detect_tokens=16, semstamp_n_max=4, translation_shots=3, article_max_words=200,
            attacks=("word_deletion",),  # DIPPER (11B) is no smoke-test model
            smoke=True,
        )
        base.update(kw)
        return cls(**base)

    @classmethod
    def from_env(cls, preset: str | None = None, **kw) -> "BenchmarkConfig":
        """Preset (explicit, else ``WMBENCH_SMOKE_TEST``) with ``WMBENCH_*`` overrides, then ``kw``.

        Recognised: WMBENCH_SMOKE_TEST (1/0), WMBENCH_MODEL, WMBENCH_ORACLE ("none" for None),
        WMBENCH_RUN_NAME, WMBENCH_VARIANTS (comma list), WMBENCH_DATASETS (comma list),
        WMBENCH_DEVICE, WMBENCH_DTYPE, WMBENCH_SEED, WMBENCH_RESULTS, WMBENCH_N_SAMPLES
        (JSON object or a single int applied to every dataset), WMBENCH_ROBUSTNESS (1/0),
        WMBENCH_ATTACKS (comma list), WMBENCH_DIPPER_MODEL.
        Explicit ``kw`` (and CLI flags, which the CLI applies afterwards) win over the environment.
        """
        env = os.environ
        if preset is None:
            smoke = _truthy(env.get("WMBENCH_SMOKE_TEST", "0"))
        elif preset in ("smoke", "full"):
            smoke = preset == "smoke"
        else:
            raise ValueError(f"unknown preset {preset!r}")
        cfg = cls.smoke_test() if smoke else cls.full()
        upd: dict = {}
        if v := env.get("WMBENCH_MODEL"):
            upd["model_name"] = v
        if (v := env.get("WMBENCH_ORACLE")) is not None:
            upd["oracle_model_name"] = None if v.strip().lower() in ("", "none", "null") else v
        if v := env.get("WMBENCH_RUN_NAME"):
            upd["run_name"] = v
        if v := env.get("WMBENCH_VARIANTS"):
            upd["variants"] = tuple(x.strip() for x in v.split(",") if x.strip())
        if v := env.get("WMBENCH_DATASETS"):
            upd["datasets"] = tuple(x.strip() for x in v.split(",") if x.strip())
        if v := env.get("WMBENCH_DEVICE"):
            upd["device"] = v
        if v := env.get("WMBENCH_DTYPE"):
            upd["dtype"] = v
        if v := env.get("WMBENCH_SEED"):
            upd["seed"] = int(v)
        if v := env.get("WMBENCH_RESULTS"):
            upd["results_root"] = v
        if v := env.get("WMBENCH_N_SAMPLES"):
            v = v.strip()
            if v.startswith("{"):
                upd["n_samples"] = {**cfg.n_samples, **json.loads(v)}
            else:
                upd["n_samples"] = {k: int(v) for k in cfg.n_samples}
        if (v := env.get("WMBENCH_ROBUSTNESS")) is not None:
            upd["run_robustness"] = _truthy(v)
        if v := env.get("WMBENCH_ATTACKS"):
            upd["attacks"] = tuple(x.strip() for x in v.split(",") if x.strip())
        if v := env.get("WMBENCH_DIPPER_MODEL"):
            upd["dipper_model"] = v
        upd.update(kw)
        return cfg.with_updates(**upd)


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def load_config_file(path: str | Path) -> BenchmarkConfig:
    """Load a JSON or YAML config file produced by ``BenchmarkConfig.as_dict``."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    for k in ("variants", "datasets", "target_fprs", "attacks"):
        if k in data and isinstance(data[k], list):
            data[k] = tuple(data[k])
    return BenchmarkConfig(**data)
