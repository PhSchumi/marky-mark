"""wmbench: benchmark MarkLLM watermarking schemes on one Hugging Face causal LM."""

from wmbench.bootstrap import (  # noqa: F401
    MARKLLM_COMMIT,
    ensure_markllm,
    ensure_nltk,
    ensure_sir_assets,
)

__version__ = "0.1.0"
