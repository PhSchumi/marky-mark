"""Environment bootstrap: pinned MarkLLM checkout, SIR assets, NLTK data.

All functions are idempotent and safe to call on an offline compute node once
``scripts/prepare_assets.py`` has been run on a node with network access.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

log = logging.getLogger("wmbench.bootstrap")

REPO_ROOT = Path(__file__).resolve().parent.parent

MARKLLM_REPO_URL = "https://github.com/THU-BPM/MarkLLM.git"
# Pinned commit (2026-09-05, "Merge pull request #75 ... textseal-watermark").
# Everything in wmbench was verified against this revision.
MARKLLM_COMMIT = "0a4fe8c642b31b8a8c4993615b1c70a0c300d918"

SIR_TRANSFORM_REPO = "Generative-Watermark-Toolkits/MarkLLM-sir"
SIR_TRANSFORM_FILE = "transform_model_cbert.pth"
SIR_EMBEDDER_REPO = "perceptiveshawty/compositional-bert-large-uncased"
SEMSTAMP_EMBEDDER_REPO = "AbeHou/SemStamp-c4-sbert"

# MarkLLM uses generic top-level package names. They must resolve to the MarkLLM
# checkout, so the checkout is inserted at the front of sys.path.
_MARKLLM_TOP_LEVEL = ("watermark", "utils", "evaluation", "exceptions", "visualize")


def default_third_party_dir() -> Path:
    return Path(os.environ.get("WMBENCH_THIRD_PARTY", REPO_ROOT / "third_party"))


def default_artifacts_dir() -> Path:
    return Path(os.environ.get("WMBENCH_ARTIFACTS", REPO_ROOT / "artifacts"))


def markllm_dir(third_party: Path | None = None) -> Path:
    return (third_party or default_third_party_dir()) / "MarkLLM"


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def _download_tarball(dest: Path) -> None:
    """Fallback when git is unavailable: fetch the pinned commit as a tarball."""
    url = f"https://github.com/THU-BPM/MarkLLM/archive/{MARKLLM_COMMIT}.tar.gz"
    tmp = dest.parent / f"markllm-{MARKLLM_COMMIT[:8]}.tar.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    urllib.request.urlretrieve(url, tmp)
    with tarfile.open(tmp) as tar:
        top = tar.getmembers()[0].name.split("/")[0]
        tar.extractall(dest.parent)
    shutil.move(str(dest.parent / top), str(dest))
    tmp.unlink(missing_ok=True)
    (dest / ".wmbench_commit").write_text(MARKLLM_COMMIT)


def ensure_markllm(third_party: Path | None = None, *, add_to_path: bool = True) -> Path:
    """Clone MarkLLM at the pinned commit if needed and put it on ``sys.path``.

    Returns the checkout directory. Set ``WMBENCH_THIRD_PARTY`` to relocate it
    (for example onto a shared filesystem on a cluster).
    """
    dest = markllm_dir(third_party)
    if (dest / "watermark" / "auto_watermark.py").exists():
        if (dest / ".git").exists():
            try:
                head = _git("rev-parse", "HEAD", cwd=dest)
                if head != MARKLLM_COMMIT:
                    log.warning("MarkLLM checkout at %s is at %s, expected %s; checking out pinned commit",
                                dest, head[:8], MARKLLM_COMMIT[:8])
                    _git("fetch", "--depth", "1", "origin", MARKLLM_COMMIT, cwd=dest)
                    _git("checkout", "--quiet", MARKLLM_COMMIT, cwd=dest)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # offline or no git
                log.warning("could not verify MarkLLM commit (%s); using existing checkout", exc)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("git"):
            log.info("cloning MarkLLM into %s", dest)
            _git("clone", "--quiet", "--filter=blob:none", MARKLLM_REPO_URL, str(dest))
            _git("checkout", "--quiet", MARKLLM_COMMIT, cwd=dest)
        else:
            _download_tarball(dest)
    if add_to_path:
        add_markllm_to_path(dest)
    return dest


def add_markllm_to_path(dest: Path) -> None:
    dest_str = str(dest)
    if dest_str in sys.path:
        sys.path.remove(dest_str)
    sys.path.insert(0, dest_str)
    os.environ.setdefault("MARKLLM_ROOT", dest_str)
    # If a foreign package with one of MarkLLM's top-level names was imported
    # earlier in the process, importing MarkLLM would silently pick it up.
    for name in _MARKLLM_TOP_LEVEL:
        mod = sys.modules.get(name)
        if mod is not None and not str(getattr(mod, "__file__", "") or "").startswith(dest_str):
            path = getattr(mod, "__path__", None)
            if not (path and any(str(p).startswith(dest_str) for p in path)):
                raise RuntimeError(
                    f"A module named {name!r} from outside MarkLLM is already imported "
                    f"({getattr(mod, '__file__', path)}). Import wmbench before anything that provides {name!r}."
                )


def ensure_nltk() -> None:
    """NLTK sentence tokenizer data (needed by SemStamp)."""
    import nltk

    for resource, probe in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(probe)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:  # offline node without the data
                log.warning("could not download NLTK %s (%s); SemStamp will not work", resource, exc)


def ensure_sir_assets(artifacts: Path | None = None) -> dict[str, str]:
    """Download the SIR transform model and embedder; re-save the checkpoint for CPU.

    The upstream ``transform_model_cbert.pth`` stores CUDA tensors, so ``torch.load``
    fails on CPU-only hosts. MarkLLM calls ``torch.load(path)`` without
    ``map_location``, therefore a CPU-mapped copy is written next to it.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    root = (artifacts or default_artifacts_dir()) / "sir"
    root.mkdir(parents=True, exist_ok=True)
    cpu_ckpt = root / "transform_model_cbert.cpu.pth"
    if not cpu_ckpt.exists():
        import torch

        raw = hf_hub_download(SIR_TRANSFORM_REPO, SIR_TRANSFORM_FILE, local_dir=root)
        state = torch.load(raw, map_location="cpu")
        torch.save(state, cpu_ckpt)
    embedder = root / "compositional-bert-large-uncased"
    if not (embedder / "config.json").exists():
        snapshot_download(SIR_EMBEDDER_REPO, local_dir=embedder,
                          allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"])
    return {"transform_model_name": str(cpu_ckpt), "embedding_model_path": str(embedder)}


def sir_mapping_path(vocab_size: int, artifacts: Path | None = None, *, seed: int = 0,
                     scale_dimension: int = 300) -> str:
    """Return a token->dimension mapping file for SIR matching ``vocab_size``.

    MarkLLM ships mappings for a few vocabularies (OPT, Llama, T5, StarCoder, NLLB)
    and otherwise creates an *unseeded* random one on first use. Here the mapping is
    created deterministically and persisted so detection stays consistent across runs.
    """
    shipped = markllm_dir() / "watermark" / "sir" / "mapping" / f"{scale_dimension}_mapping_{vocab_size}.json"
    if shipped.exists():
        return str(shipped)
    root = (artifacts or default_artifacts_dir()) / "sir"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{scale_dimension}_mapping_{vocab_size}_seed{seed}.json"
    if not path.exists():
        rng = random.Random(seed)
        mapping = [rng.randint(0, scale_dimension - 1) for _ in range(vocab_size)]
        path.write_text(json.dumps(mapping))
    return str(path)


def ensure_semstamp_assets() -> str:
    """Pre-download the SemStamp sentence embedder into the HF cache; returns the repo id."""
    from huggingface_hub import snapshot_download

    snapshot_download(SEMSTAMP_EMBEDDER_REPO, allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"])
    return SEMSTAMP_EMBEDDER_REPO


def prefetch_models(model_names: list[str]) -> None:
    """Populate the HF cache (weights + tokenizer) so compute nodes can run offline."""
    from huggingface_hub import snapshot_download

    for name in model_names:
        if name:
            log.info("prefetching %s", name)
            snapshot_download(name, allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin",
                                                    "*.tiktoken", "*.py"])
