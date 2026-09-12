"""Model loading, device selection and prompt formatting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger("wmbench.models")


def pick_device(preference: str | None = "auto") -> str:
    """Resolve ``"auto"`` to ``cuda`` > ``mps`` > ``cpu``."""
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, preference: str | None = "auto") -> torch.dtype:
    """fp16 on CUDA (bf16 if requested and supported), fp32 elsewhere."""
    if preference and preference != "auto":
        return getattr(torch, preference)
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _from_pretrained_dtype_kwarg(dtype: torch.dtype) -> dict:
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def load_tokenizer(name: str, *, trust_remote_code: bool = False) -> PreTrainedTokenizerBase:
    """Tokenizer with a pad token (eos) if it has none; MarkLLM's PF decoder needs one."""
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


_GENERATION_CONFIG_KEEP = {"bos_token_id", "eos_token_id", "pad_token_id", "transformers_version", "_from_model_config"}


def neutralize_generation_config(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> dict:
    """Replace the hub ``generation_config`` by token ids only; return the discarded fields.

    Hub checkpoints ship defaults such as ``repetition_penalty`` or ``no_repeat_ngram_size`` that
    ``model.generate`` applies silently. The HF-generate variants would inherit them while PF's own
    decoding loop would not, breaking the "one base distribution" contract, so everything except the
    token ids is dropped (and logged).
    """
    old = model.generation_config
    eos = old.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)
    dropped = {k: v for k, v in old.to_diff_dict().items() if k not in _GENERATION_CONFIG_KEEP}
    model.generation_config = GenerationConfig(
        bos_token_id=old.bos_token_id if old.bos_token_id is not None else tokenizer.bos_token_id,
        eos_token_id=eos_ids or None, pad_token_id=tokenizer.pad_token_id)
    if dropped:
        log.warning("discarded hub generation_config fields %s so that every variant samples from the same base "
                    "distribution", dropped)
    return dropped


def stop_token_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """All ids that end a generation (every EOS id of the model plus the tokenizer's EOS/pad)."""
    eos = model.generation_config.eos_token_id
    ids = set([eos] if isinstance(eos, int) else list(eos or []))
    for t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if t is not None:
            ids.add(int(t))
    return ids


def load_causal_lm(name: str, device: str, dtype: torch.dtype, *, trust_remote_code: bool = False,
                   tokenizer_name: str | None = None,
                   tokenizer: PreTrainedTokenizerBase | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM (and tokenizer unless given) onto ``device`` in eval mode.

    The hub generation config is neutralised (see :func:`neutralize_generation_config`).
    """
    tok = tokenizer or load_tokenizer(tokenizer_name or name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=trust_remote_code, **_from_pretrained_dtype_kwarg(dtype)
    )
    model = model.to(device).eval()
    neutralize_generation_config(model, tok)
    log.info("loaded %s on %s (%s), %.1fM params", name, device, dtype,
             sum(p.numel() for p in model.parameters()) / 1e6)
    return model, tok


def max_positions(model: PreTrainedModel, default: int = 2048) -> int:
    """Context window of the model (``max_position_embeddings`` or a sensible fallback)."""
    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return default


def logits_vocab_size(model: PreTrainedModel) -> int:
    """Size of the logit vector. This, not ``len(tokenizer)``, must be passed to MarkLLM.

    Greenlist permutations (KGW, MorphMark), SIR's token mapping and PF's random
    vectors are all built with ``vocab_size`` and indexed against the logits, so the
    two must agree. For OPT the tokenizer has 50265 entries but the model emits 50272 logits.
    """
    emb = model.get_output_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return int(emb.weight.shape[0])
    return int(model.config.vocab_size)


@dataclass
class PromptFormatter:
    """Turn a task prompt into the token ids fed to the model.

    Every variant consumes the *same* ids, so BOS handling is identical across
    variants. For chat/instruct models the tokenizer's chat template is applied with
    the generation prompt appended; base models get ``add_special_tokens=True``
    (a BOS for OPT/Llama-style tokenizers, matching MarkLLM's own generation path).
    """

    tokenizer: PreTrainedTokenizerBase
    use_chat_template: bool | None = None  # None = auto-detect
    system_prompt: str | None = None

    @property
    def is_chat(self) -> bool:
        if self.use_chat_template is not None:
            return self.use_chat_template
        return bool(getattr(self.tokenizer, "chat_template", None))

    def format(self, user_text: str) -> str:
        """Prompt text (for logging and for text-based algorithms such as SemStamp)."""
        if not self.is_chat:
            return user_text
        messages = self._messages(user_text)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode(self, user_text: str) -> list[int]:
        if not self.is_chat:
            return list(self.tokenizer(user_text, add_special_tokens=True)["input_ids"])
        ids = self.tokenizer.apply_chat_template(self._messages(user_text), tokenize=True,
                                                 add_generation_prompt=True)
        if hasattr(ids, "input_ids"):  # BatchEncoding in some transformers versions
            ids = ids["input_ids"]
        return list(ids)

    def _messages(self, user_text: str) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages


def extract_continuation(full_text: str, prompt: str) -> str:
    """Return the generated part of a decoded ``prompt + continuation`` string.

    Only needed for text-based algorithms (SemStamp) that return the full text;
    token-level variants slice ids instead. When tokenization round-trips alter
    whitespace, fall back to MarkLLM's word-count truncation.
    """
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    stripped = prompt.strip()
    if stripped and full_text.lstrip().startswith(stripped):
        return full_text.lstrip()[len(stripped):]
    n_words = len(prompt.split())
    return " ".join(full_text.split()[n_words:])
