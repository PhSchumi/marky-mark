"""Watermark variants: one generate/detect interface over the MarkLLM algorithms.

Every variant consumes the same prompt token ids and the same sampling settings
(:data:`wmbench.config.SAMPLING`). Where MarkLLM's own generation path would break
that equality it is bypassed or patched here; each deviation is documented on the
builder that applies it.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessorList

from wmbench.bootstrap import (
    REPO_ROOT,
    default_artifacts_dir,
    ensure_markllm,
    ensure_nltk,
    ensure_sir_assets,
    sir_mapping_path,
)
from wmbench.config import SAMPLING
from wmbench.models import extract_continuation, logits_vocab_size, max_positions

log = logging.getLogger("wmbench.watermarks")

CONFIG_DIR = REPO_ROOT / "configs" / "watermarks"


def _one_sided_normal_tail(z: float) -> float:
    return float(0.5 * math.erfc(z / math.sqrt(2)))


@dataclass(frozen=True)
class VariantSpec:
    name: str
    algorithm: str | None  # MarkLLM algorithm key, None for the unwatermarked baseline
    config_file: str | None  # under configs/watermarks/
    family: str
    description: str
    score_name: str
    fixed_threshold: float | None
    nominal_fpr: float | None  # FPR implied by the shipped threshold under the algorithm's null model
    citation: str = ""
    datasets: tuple[str, ...] | None = None  # None = every dataset
    notes: str = ""


VARIANTS: dict[str, VariantSpec] = {
    "none": VariantSpec(
        "none", None, None, "baseline",
        "Unwatermarked multinomial sampling from the same model, prompt ids and sampling settings.",
        "-", None, None),
    "KGW": VariantSpec(
        "KGW", "KGW", "KGW.json", "logit bias (green list)",
        "Hashes the previous token to seed a pseudo-random 'green list' of gamma*|V| tokens and adds "
        "delta to their logits. Detection counts green tokens and reports a one-proportion z-score.",
        "z-score (green-token count)", 4.0, _one_sided_normal_tail(4.0),
        "Kirchenbauer et al., ICML 2023, arXiv:2301.10226"),
    "MorphMark": VariantSpec(
        "MorphMark", "MorphMark", "MorphMark.json", "adaptive probability shift (green list)",
        "Same green list as KGW, but instead of a fixed logit bias it moves probability mass to the "
        "green list as a function of the mass it already has (linear/exp/log 'morph' curve), so "
        "low-entropy steps are left almost untouched. Detection is KGW's z-score.",
        "z-score (green-token count)", 2.0, _one_sided_normal_tail(2.0),
        "Wang et al., ACL 2025, arXiv:2505.11541"),
    "PF": VariantSpec(
        "PF", "PF", "PF.json", "keyed decoder (Permute-and-Flip)",
        "Replaces sampling with the Permute-and-Flip decoder: the next token is argmax of "
        "log p(token) - log r(token) where r is a pseudo-random vector keyed by the previous n-gram. "
        "Detection sums -log r over the tokens actually emitted; under no watermark the sum is Gamma(k, 1).",
        "-log10 p-value (Gamma tail of the summed score)", 2.0, 0.01,
        "Zhao, Li and Wang, ICLR 2025, arXiv:2402.05864",
        notes="Score reported by wmbench is -log10 of the Gamma-tail p-value with k = number of scored tokens; "
              "MarkLLM's raw sum grows with length and its threshold subtracts the n-gram size twice."),
    "SynthID": VariantSpec(
        "SynthID", "SynthID", "SynthID.json", "tournament sampling",
        "SynthID-Text: pseudo-random g-values keyed by the previous n-gram and 30 layered keys drive a "
        "tournament that reweights the sampling distribution (non-distortionary mode, repeated contexts "
        "are skipped). Detection is the mean g-value of the text's n-grams (0.5 expected without watermark).",
        "mean g-value", 0.52, None,
        "Dathathri et al., Nature 2024, doi:10.1038/s41586-024-08025-4",
        notes="MarkLLM never resets the processor's context history between prompts; wmbench resets it."),
    "SIR": VariantSpec(
        "SIR", "SIR", "SIR.json", "semantic logit bias",
        "Semantic Invariant Robust watermark: the bias added to each token's logit is produced by a "
        "trained transform network from the sentence embedding of the preceding text, so paraphrases "
        "with the same meaning get the same watermark. Detection averages the bias sign over tokens.",
        "mean token bias similarity", 0.2, None,
        "Liu et al., ICLR 2024, arXiv:2310.06356",
        notes="Needs the compositional-BERT embedder and the transform network (downloaded by prepare_assets)."),
    "SemStamp": VariantSpec(
        "SemStamp", "SEMSTAMP", "SEMSTAMP.json", "sentence-level rejection sampling",
        "Generates sentence by sentence and rejects sentences whose sentence-embedding LSH bucket is "
        "not in the 'valid' set seeded by the previous sentence (up to N_max tries). Detection counts "
        "valid sentences and reports a z-score. Semantic, paraphrase-robust, and expensive.",
        "z-score (valid-sentence count)", 3.0, _one_sided_normal_tail(3.0),
        "Hou et al., NAACL 2024, arXiv:2310.03991",
        datasets=("c4",),
        notes="Optional; up to N_max generations per sentence. wmbench lifts MarkLLM's OPT-only check "
              "and uses the shared sampling settings instead of the hard-coded temperature 0.7."),
}

WATERMARK_VARIANTS = tuple(k for k in VARIANTS if k != "none")


@dataclass
class GenerationOutput:
    continuation_ids: list[int]
    continuation: str
    prompt_text: str | None = None


@dataclass
class DetectionOutput:
    score: float
    is_watermarked: bool | None
    n_tokens: int
    error: str | None = None
    raw: dict = field(default_factory=dict)


class WatermarkVariant:
    """A model + one watermarking scheme (or none), with token-level generation."""

    def __init__(self, spec: VariantSpec, model, tokenizer, device: str, max_new_tokens: int, *,
                 overrides: dict | None = None, artifacts: Path | None = None, semstamp_n_max: int = 100):
        self.spec = spec
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_new_tokens = max_new_tokens
        self.overrides = dict(overrides or {})
        self.artifacts = artifacts or default_artifacts_dir()
        self.semstamp_n_max = semstamp_n_max
        self.vocab_size = logits_vocab_size(model)
        self.max_positions = max_positions(model)
        self.gen_kwargs = dict(SAMPLING, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        self.wm = None
        self._processor = None
        if spec.algorithm:
            self._build()

    # ------------------------------------------------------------------ construction
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def is_watermarked(self) -> bool:
        return self.spec.algorithm is not None

    def effective_config(self) -> dict:
        cfg = dict(getattr(getattr(self.wm, "config", None), "config_dict", {}) or {})
        cfg["sampling"] = dict(SAMPLING, max_new_tokens=self.max_new_tokens)
        return cfg

    def _build(self) -> None:
        ensure_markllm()
        from utils.transformers_config import TransformersConfig
        from watermark.auto_watermark import AutoWatermark

        alg = self.spec.algorithm
        cfg_path = CONFIG_DIR / self.spec.config_file
        if not cfg_path.exists():
            raise FileNotFoundError(f"watermark config {cfg_path} missing")
        tc = TransformersConfig(model=self.model, tokenizer=self.tokenizer, vocab_size=self.vocab_size,
                                device=self.device, **self.gen_kwargs)
        # SynthIDConfig reads these via getattr on the TransformersConfig object, not gen_kwargs.
        tc.temperature = SAMPLING["temperature"]
        tc.top_k = 0  # top_k > 0 would make SynthID return [batch, top_k]-shaped scores and crash
        overrides = dict(self.overrides)

        if alg == "SIR":
            import jieba

            jieba.setLogLevel(logging.WARNING)  # jieba logs at DEBUG by default
            overrides.update(ensure_sir_assets(self.artifacts))
            overrides["mapping_name"] = sir_mapping_path(self.vocab_size, self.artifacts)
            mapping = json.loads(Path(overrides["mapping_name"]).read_text())
            if len(mapping) != self.vocab_size:
                raise ValueError(f"SIR mapping {overrides['mapping_name']} has {len(mapping)} entries, "
                                 f"model emits {self.vocab_size} logits")
            for key in ("transform_model_name", "embedding_model_path"):
                if not Path(overrides[key]).exists():
                    raise FileNotFoundError(f"SIR asset missing: {overrides[key]}")
        elif alg == "PF":
            overrides["max_seq_len"] = self.max_positions
        elif alg == "SEMSTAMP":
            ensure_nltk()
            overrides.update(max_new_tokens=self.max_new_tokens, min_new_tokens=self.max_new_tokens,
                             N_max=self.semstamp_n_max)

        if alg == "PF":
            from watermark.pf import PFConfig

            self.wm = _PFWithIds(PFConfig(str(cfg_path), tc, **overrides))
        elif alg == "SEMSTAMP":
            from watermark.semstamp import SemStampConfig

            self.wm = _SemStampGeneral(SemStampConfig(str(cfg_path), tc, **overrides))
        else:
            self.wm = AutoWatermark.load(alg, algorithm_config=str(cfg_path), transformers_config=tc, **overrides)
            self._processor = self.wm.logits_processor
        log.info("built variant %s (%s) max_new_tokens=%d", self.name, alg, self.max_new_tokens)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def _hf_generate(self, prompt_ids: list[int], processor=None) -> list[int]:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        kwargs = dict(self.gen_kwargs)
        if processor is not None:
            kwargs["logits_processor"] = LogitsProcessorList([processor])
        out = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **kwargs)
        ids = out[0, input_ids.shape[1]:].tolist()
        stop = {t for t in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id) if t is not None}
        while ids and ids[-1] in stop:
            ids.pop()
        return ids

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate(self, prompt_ids: list[int], prompt_text: str | None = None) -> GenerationOutput:
        """Generate a continuation for ``prompt_ids``; returns continuation ids and text."""
        if len(prompt_ids) + self.max_new_tokens > self.max_positions:
            raise ValueError(f"prompt ({len(prompt_ids)} tokens) + max_new_tokens ({self.max_new_tokens}) "
                             f"exceeds the model context ({self.max_positions})")
        alg = self.spec.algorithm
        if alg is None:
            ids = self._hf_generate(prompt_ids)
        elif alg == "SynthID":
            self._processor.state = None  # MarkLLM carries context history across prompts; reset it
            ids = self._hf_generate(prompt_ids, self._processor)
        elif alg == "PF":
            ids = self.wm.generate_from_ids(prompt_ids)
        elif alg == "SEMSTAMP":
            text = prompt_text if prompt_text is not None else self._decode(prompt_ids)
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                full = self.wm.generate_watermarked_text(text)
            for line in buf.getvalue().splitlines():
                if line.startswith("WARNING"):
                    log.debug("SemStamp: %s", line)
            cont = extract_continuation(full, text)
            ids = self.tokenizer(cont, add_special_tokens=False)["input_ids"]
            return GenerationOutput(list(ids), cont, text)
        else:  # KGW, MorphMark, SIR: plain HF generate with the algorithm's logits processor
            ids = self._hf_generate(prompt_ids, self._processor)
        return GenerationOutput(ids, self._decode(ids), prompt_text)

    # ------------------------------------------------------------------ detection
    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def detect(self, text: str) -> DetectionOutput:
        """Score a continuation. Failures (too short, empty) become ``nan`` with the error recorded."""
        if not self.is_watermarked:
            raise RuntimeError("the 'none' variant has no detector")
        n = self.n_tokens(text)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.spec.algorithm == "PF":
                    return self._detect_pf(text, n)
                res = self.wm.detect_watermark(text, return_dict=True)
            score = float(res["score"])
            flag = res.get("is_watermarked")
            flag = None if flag is None else bool(flag)
            if not math.isfinite(score):
                return DetectionOutput(float("nan"), None, n, "non-finite score", {})
            return DetectionOutput(score, flag, n, None, {k: _jsonable(v) for k, v in res.items()})
        except Exception as exc:  # noqa: BLE001 - any detector failure is recorded, not fatal
            return DetectionOutput(float("nan"), None, n, f"{type(exc).__name__}: {exc}", {})

    def _detect_pf(self, text: str, n: int) -> DetectionOutput:
        from scipy.stats import gamma

        per_token = self.wm.utils.get_scores_by_t(text)  # shape (k, 1): -log r for each scored token
        k = int(len(per_token))
        if k == 0:
            return DetectionOutput(float("nan"), None, n, "fewer than ngram+2 tokens", {})
        total = float(np.asarray(per_token).sum())
        p = float(gamma.sf(total, a=k))
        score = -math.log10(max(p, 1e-300))
        return DetectionOutput(score, p < 0.01, n, None,
                               {"raw_sum": total, "n_scored": k, "p_value": p})


def _jsonable(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def build_variant(name: str, model, tokenizer, device: str, max_new_tokens: int, **kw) -> WatermarkVariant:
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {list(VARIANTS)}")
    return WatermarkVariant(VARIANTS[name], model, tokenizer, device, max_new_tokens, **kw)


# ============================================================================ MarkLLM patches
# The classes below subclass MarkLLM (Apache-2.0, THU-BPM) and re-implement one method each.
# They are created lazily because MarkLLM is only importable after ensure_markllm().


@functools.lru_cache(maxsize=None)
def _pf_with_ids_class():
    ensure_markllm()
    from watermark.pf import PF

    return type("PFWithIds", (PF,), {"generate_from_ids": _pf_generate_from_ids,
                                     "__doc__": _pf_generate_from_ids.__doc__})


@functools.lru_cache(maxsize=None)
def _semstamp_general_class():
    ensure_markllm()
    from watermark.semstamp import SemStamp

    return type("SemStampGeneral", (SemStamp,), {"generate_watermarked_text": _semstamp_generate,
                                                 "__doc__": _semstamp_generate.__doc__})


def _PFWithIds(config):
    return _pf_with_ids_class()(config)


def _SemStampGeneral(config):
    return _semstamp_general_class()(config)


@torch.no_grad()
def _pf_generate_from_ids(self, prompt_ids: list[int]) -> list[int]:
    """MarkLLM PF decoder taking prompt token ids.

    Differences from ``PF.generate_watermarked_text`` (watermark/pf/pf.py):
    * consumes the shared prompt ids (BOS handled like every other variant) instead of
      re-encoding the prompt with ``add_special_tokens=False``;
    * casts the last-step logits to fp32 before Permute-and-Flip sampling;
    * stops at EOS instead of filling the buffer and truncating afterwards;
    * returns continuation ids only (no special tokens leak into the decoded text).
    """
    cfg = self.config
    model = cfg.generation_model
    max_new = int(cfg.gen_kwargs["max_new_tokens"])
    total_len = len(prompt_ids) + max_new
    if total_len > cfg.max_seq_len:
        raise ValueError(f"PF: prompt {len(prompt_ids)} + max_new_tokens {max_new} > max_seq_len {cfg.max_seq_len}")
    temperature = float(cfg.gen_kwargs.get("temperature", 1.0))
    top_p = float(cfg.gen_kwargs.get("top_p", 1.0))
    dev = model.device
    tokens = torch.full((total_len,), self.utils.pad_id, dtype=torch.long, device=dev)
    tokens[: len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=dev)
    past, prev_pos, end = None, 0, len(prompt_ids)
    for cur_pos in range(len(prompt_ids), total_len):
        outputs = model(tokens[prev_pos:cur_pos].unsqueeze(0), use_cache=True, past_key_values=past)
        past = outputs.past_key_values
        ngram_tokens = tokens[max(0, cur_pos - cfg.ngram): cur_pos]
        logits = outputs.logits[:, -1, :].float()
        next_tok = self.utils.sample_next(logits, ngram_tokens, temperature, top_p)
        tok = int(next_tok.reshape(-1)[0].item())
        if tok == self.utils.eos_id:
            break
        tokens[cur_pos] = tok
        end = cur_pos + 1
        prev_pos = cur_pos
    return tokens[len(prompt_ids):end].tolist()


def _semstamp_generate(self, prompt: str, *args, **kwargs) -> str:  # noqa: C901 - upstream loop
    """MarkLLM SemStamp loop without the OPT-only check and with the shared sampling settings.

    ``watermark/semstamp/semstamp.py`` with three changes: ``model.generate`` gets
    ``config.gen_kwargs`` (plus ``min_new_tokens`` so EOS cannot cut a sentence short; the
    sentence-end stopping criterion controls length) instead of hard-coded OPT settings;
    any model works; after ``N_max`` rejected candidates the last candidate is accepted
    (the paper's fallback; MarkLLM's loop never accepts an invalid sentence).
    """
    import torch.nn.functional as F
    from transformers import StoppingCriteriaList

    cfg = self.config
    utils = self.utils
    lsh_model = utils.get_lsh_model()
    sent_end_criteria = utils.SentenceEndCriteria(cfg.generation_tokenizer)
    gen_kwargs = dict(cfg.gen_kwargs)
    gen_kwargs["max_new_tokens"] = cfg.max_new_tokens
    gen_kwargs["min_new_tokens"] = cfg.min_new_tokens

    lsh_seed = lsh_model.get_hash([prompt])[0]
    n_bins = 2 ** cfg.dimension_d
    n_accept = int(n_bins * cfg.gamma)
    utils.rng.manual_seed(cfg.prime_P * lsh_seed)
    accept_mask = torch.randperm(n_bins, device=cfg.device, generator=utils.rng)[:n_accept]

    text = prompt
    text_ids = cfg.generation_tokenizer.encode(prompt, return_tensors="pt").to(cfg.device)
    prompt_length = text_ids.shape[1]
    sent_end_criteria.update(text)
    current_trials = 0
    while True:
        outputs = cfg.generation_model.generate(
            text_ids, stopping_criteria=StoppingCriteriaList([sent_end_criteria]), **gen_kwargs)
        new_text_ids = outputs[:, :-1]
        new_text = cfg.generation_tokenizer.decode(new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
        if new_text == "":
            break
        current_trials += 1
        embeds = torch.tensor(lsh_model.get_embeddings([new_text]), device=cfg.device)
        normals = torch.tensor(lsh_model.hasher.normals, device=cfg.device, dtype=embeds.dtype)
        sims = F.cosine_similarity(embeds.expand(normals.size(0), -1), normals)
        if sims.abs().min().item() < cfg.margin_m and current_trials < cfg.N_max:
            continue  # too close to an LSH hyperplane: not robust, resample
        lsh_candidate = lsh_model.get_hash([new_text])[0]
        if lsh_candidate not in accept_mask and current_trials < cfg.N_max:
            continue
        if current_trials >= cfg.N_max:
            log.debug("SemStamp: no valid sentence after %d trials; keeping last candidate", cfg.N_max)
        current_trials = 0
        lsh_seed = lsh_candidate
        utils.rng.manual_seed(cfg.prime_P * lsh_seed)
        accept_mask = torch.randperm(n_bins, device=cfg.device, generator=utils.rng)[:n_accept]
        text += new_text
        text_ids = new_text_ids.to(cfg.device)
        sent_end_criteria.update(text)
        if text_ids.shape[1] - prompt_length >= cfg.max_new_tokens - 1:
            break
    return text
