"""Text-level robustness attacks (optional track, stage ``attack``).

* ``word_deletion``: MarkLLM's ``WordDeletion`` rule (every whitespace-delimited word is dropped
  independently with probability ``ratio``), seeded per call.
* ``dipper``: the DIPPER paraphraser (Krishna et al., 2023). Same procedure and defaults as MarkLLM's
  ``evaluation/tools/text_editor.py::DipperParaphraser``: the text is paraphrased in windows of
  ``sent_interval`` sentences, each conditioned on the prompt plus the paraphrase produced so far, with
  the control codes ``lexical = 100 - lex_diversity, order = 100 - order_diversity``. Re-implemented
  here so the benchmark does not import MarkLLM's evaluation module (it pulls in the ``translate`` and
  OpenAI clients at import time).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import random
from dataclasses import dataclass

import torch

from wmbench.bootstrap import ensure_nltk

log = logging.getLogger("wmbench.attacks")


@dataclass(frozen=True)
class AttackSpec:
    name: str
    description: str
    needs_model: bool
    citation: str = ""


ATTACKS: dict[str, AttackSpec] = {
    "word_deletion": AttackSpec(
        "word_deletion", "Delete each word independently with probability `word_deletion_ratio`.", False,
        "MarkLLM WordDeletion editor; Kirchenbauer et al., ICML 2023 (deletion attacks)"),
    "dipper": AttackSpec(
        "dipper", "DIPPER paraphrase (T5-XXL): sentence windows re-written with lexical/order diversity codes, "
                  "conditioned on the prompt and the paraphrase so far.", True,
        "Krishna et al., NeurIPS 2023, arXiv:2303.13408"),
}

DIVERSITY_LEVELS = (0, 20, 40, 60, 80, 100)
DIPPER_GEN_KWARGS = {"do_sample": True, "top_p": 0.75, "top_k": 0, "max_new_tokens": 512}


def word_deletion(text: str, ratio: float, seed: int) -> str:
    """Delete each whitespace-delimited word independently with probability ``ratio``.

    Same rule as MarkLLM's ``WordDeletion`` editor, but seeded per call so resumed or
    array-split runs reproduce the same attacked text.
    """
    if not text:
        return text
    rng = random.Random(seed)
    kept = [w for w in text.split() if rng.random() >= ratio]
    return " ".join(kept)


class Attacker:
    """One attack: ``attack(text, prompt, seed) -> text``; ``config()`` identifies it for hashing."""

    name: str = ""

    def config(self) -> dict:
        raise NotImplementedError

    def attack(self, text: str, prompt: str, seed: int) -> str:
        raise NotImplementedError

    def close(self) -> None:
        """Release model memory (no-op for text-only attacks)."""


class WordDeletionAttacker(Attacker):
    name = "word_deletion"

    def __init__(self, ratio: float):
        self.ratio = float(ratio)

    def config(self) -> dict:
        return {"attack": self.name, "ratio": self.ratio}

    def attack(self, text: str, prompt: str, seed: int) -> str:
        return word_deletion(text, self.ratio, seed)


class DipperAttacker(Attacker):
    """DIPPER paraphraser over a loaded seq2seq model (see the module docstring for the procedure)."""

    name = "dipper"

    def __init__(self, model, tokenizer, *, model_name: str, tokenizer_name: str, revision: str | None = None,
                 lex_diversity: int = 60, order_diversity: int = 0, sent_interval: int = 1,
                 gen_kwargs: dict | None = None):
        for value, label in ((lex_diversity, "lexical"), (order_diversity, "order")):
            if value not in DIVERSITY_LEVELS:
                raise ValueError(f"{label} diversity must be one of {DIVERSITY_LEVELS}, got {value!r}")
        if sent_interval < 1:
            raise ValueError("sent_interval must be >= 1")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name, self.tokenizer_name, self.revision = model_name, tokenizer_name, revision
        self.lex_diversity, self.order_diversity, self.sent_interval = int(lex_diversity), int(order_diversity), int(sent_interval)
        self.gen_kwargs = dict(DIPPER_GEN_KWARGS, **(gen_kwargs or {}))

    @classmethod
    def from_pretrained(cls, cfg, device: str) -> "DipperAttacker":
        """Load ``cfg.dipper_model`` / ``cfg.dipper_tokenizer`` onto ``device`` (bf16 on CUDA when supported)."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from wmbench.models import _from_pretrained_dtype_kwarg

        dtype = dipper_dtype(device, cfg.dipper_dtype)
        log.info("loading DIPPER %s (%s) on %s", cfg.dipper_model, dtype, device)
        tokenizer = AutoTokenizer.from_pretrained(cfg.dipper_tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.dipper_model, **_from_pretrained_dtype_kwarg(dtype)).to(device)
        return cls(model, tokenizer, model_name=cfg.dipper_model, tokenizer_name=cfg.dipper_tokenizer,
                   revision=getattr(model.config, "_commit_hash", None), lex_diversity=cfg.dipper_lex_diversity,
                   order_diversity=cfg.dipper_order_diversity, sent_interval=cfg.dipper_sent_interval)

    def config(self) -> dict:
        return {"attack": self.name, "model": self.model_name, "tokenizer": self.tokenizer_name,
                "revision": self.revision, "lex_diversity": self.lex_diversity,
                "order_diversity": self.order_diversity, "sent_interval": self.sent_interval,
                "gen_kwargs": self.gen_kwargs}

    def attack(self, text: str, prompt: str, seed: int) -> str:
        text = " ".join((text or "").split())
        if not text:
            return ""
        ensure_nltk()
        from nltk.tokenize import sent_tokenize

        torch.manual_seed(seed)  # seeds CPU and every CUDA device
        sentences = sent_tokenize(text)
        prefix = " ".join((prompt or "").replace("\n", " ").split())
        lex_code, order_code = 100 - self.lex_diversity, 100 - self.order_diversity
        pieces = []
        for i in range(0, len(sentences), self.sent_interval):
            window = " ".join(sentences[i:i + self.sent_interval])
            inp = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                inp += f" {prefix}"
            inp += f" <sent> {window} </sent>"
            enc = self.tokenizer([inp], return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.inference_mode():
                out_ids = self.model.generate(**enc, **self.gen_kwargs)
            piece = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            prefix = f"{prefix} {piece}".strip()
            pieces.append(piece)
        return " ".join(pieces).strip()

    def close(self) -> None:
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def dipper_dtype(device: str, preference: str = "auto") -> torch.dtype:
    """T5 overflows in fp16; use bf16 on CUDA when the device supports it, fp32 elsewhere."""
    if preference and preference != "auto":
        return getattr(torch, preference)
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        log.warning("CUDA device without bf16: loading DIPPER in fp16 (T5 may overflow; set dipper_dtype=float32)")
        return torch.float16
    return torch.float32


def load_similarity_scorer(model_name: str, device: str):
    """``(source, attacked) -> cosine similarity`` of sentence embeddings; ``None`` if the model cannot load."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:  # noqa: BLE001 - offline node without the model: strength is reported without it
        log.warning("similarity model %s unavailable (%s); sim_embed will be null", model_name, exc)
        return None

    def score(source: str, attacked: str) -> float:
        emb = model.encode([source, attacked], convert_to_tensor=True, normalize_embeddings=True)
        return float((emb[0] * emb[1]).sum().item())

    return score


def build_attacker(name: str, cfg, device: str) -> Attacker:
    if name not in ATTACKS:
        raise KeyError(f"unknown attack {name!r}; known: {list(ATTACKS)}")
    if name == "word_deletion":
        return WordDeletionAttacker(cfg.word_deletion_ratio)
    return DipperAttacker.from_pretrained(cfg, device)


def attack_config_digest(attacker: Attacker) -> str:
    """Hash of everything that determines the attacked text (attack, model, revision, settings)."""
    return hashlib.sha1(json.dumps(attacker.config(), sort_keys=True, default=str).encode()).hexdigest()[:12]
