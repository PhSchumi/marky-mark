#!/usr/bin/env python
"""Generate notebooks/watermark_benchmark.ipynb from code (keeps the notebook valid and diffable).

Run ``python scripts/build_notebook.py`` after editing this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "watermark_benchmark.ipynb"

cells = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text: str, tags: list[str] | None = None) -> None:
    c = new_code_cell(text.strip("\n"))
    if tags:
        c.metadata["tags"] = tags
    cells.append(c)


# --------------------------------------------------------------------------- intro
md("""
# Watermarking one LLM six ways: MarkLLM watermarks compared

This notebook takes **one Hugging Face causal language model** (a placeholder you replace), wraps it with
several text watermarking schemes from the [MarkLLM](https://github.com/THU-BPM/MarkLLM) toolkit, and
compares the resulting *watermarked models* on

* **detectability** - can the watermark be found in the generated text (true positive rate at a fixed
  false positive rate against human text, AUROC, F1)?
* **text quality** - prompt-conditioned perplexity under an oracle model and n-gram diversity on C4 news continuations;
* **standard benchmarks** - BLEU on WMT16 German->English, ROUGE on CNN/DailyMail summarisation, and
  optionally pass@1 on HumanEval;
* **cost** - tokens per second;
* optionally **robustness** to a word-deletion attack.

Variants (the "versions" of the model):

| variant | family | paper |
|---|---|---|
| `none` | unwatermarked baseline, same sampling | - |
| `KGW` | green-list logit bias (reference method) | Kirchenbauer et al., ICML 2023 |
| `MorphMark` | adaptive green-list probability shift | Wang et al., ACL 2025 |
| `PF` | Permute-and-Flip keyed decoder | Zhao, Li & Wang, ICLR 2025 |
| `SynthID` | SynthID-Text tournament sampling | Dathathri et al., Nature 2024 |
| `SIR` | Semantic Invariant Robust logit bias | Liu et al., ICLR 2024 |
| `SemStamp` (optional) | sentence-level semantic rejection sampling | Hou et al., NAACL 2024 |

All heavy lifting is in the `wmbench` package of this repository; the notebook is a narrated driver
that calls the same functions as `scripts/run_benchmark.py`. Every stage caches its records under
`results/<run_name>/`, so re-running a cell after an interruption continues where it stopped.

**How to run**

* Interactive: run top to bottom. The default is a **smoke test** (`facebook/opt-125m`, a handful of
  samples, CPU-friendly) that only checks the pipeline. Set `SMOKE_TEST = False` (or the environment
  variable `WMBENCH_SMOKE_TEST=0`) for the real run on a GPU.
* Headless on a cluster: `sbatch scripts/slurm/notebook.sbatch` (uses `jupyter nbconvert --execute`
  with no cell timeout), after `python scripts/prepare_assets.py` on a node with internet.
* Large runs: prefer the CLI + SLURM array in `scripts/slurm/` and use section 7 of this notebook only
  to read the results.
""")

# --------------------------------------------------------------------------- setup
md("""
## 0. Setup

MarkLLM is **not** installed from PyPI (the wheel is stale and lacks MorphMark, PF, SynthID and SemStamp);
`wmbench.bootstrap.ensure_markllm()` clones it at a pinned commit into `third_party/MarkLLM` and puts it on
`sys.path`. SIR additionally needs two downloads (a transform network and a BERT embedder); they are fetched
on first use or by `scripts/prepare_assets.py`.
""")
code("""
# Fresh environment? Uncomment (torch first, matching your CUDA):
# %pip install -r ../requirements.txt

import os, sys, logging, json, time
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "wmbench").exists():          # notebook launched from notebooks/
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "huggingface_hub", "urllib3", "jieba"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

import torch, transformers, pandas as pd
from IPython.display import Image, Markdown, display

import wmbench
from wmbench.bootstrap import ensure_markllm, MARKLLM_COMMIT
from wmbench.config import BenchmarkConfig, SAMPLING
from wmbench.evaluate import Runner
from wmbench.watermarks import VARIANTS, build_variant, CONFIG_DIR

markllm_root = ensure_markllm()
print(f"wmbench {wmbench.__version__} | MarkLLM {MARKLLM_COMMIT[:8]} at {markllm_root}")
print(f"torch {torch.__version__} | transformers {transformers.__version__} | cuda={torch.cuda.is_available()}")
""")

# --------------------------------------------------------------------------- config
md("""
## 1. Configuration

Everything you would normally edit is in the next cell. The placeholder model is `facebook/opt-1.3b`
(MarkLLM's reference model; the SIR vocabulary mapping for it ships with MarkLLM). Any Hugging Face causal LM
works; chat/instruct models are detected from the tokenizer's chat template and prompted through it.

`SMOKE_TEST` switches between the two presets:

| | smoke | full |
|---|---|---|
| model | `facebook/opt-125m` | `MODEL_NAME` |
| PPL oracle | the model itself | `ORACLE_MODEL` (`facebook/opt-2.7b`, i.e. a larger model of the same family) |
| C4 / WMT16 / CNN-DM samples | 8 / 6 / 4 | 500 / 200 / 100 |
| new tokens (C4 / WMT16 / CNN-DM) | 64 / 32 / 48 | 200 / 64 / 96 |

Sampling is fixed to plain multinomial sampling (`temperature=1, top_p=1, top_k=0`) for **every** variant:
it is the only setting under which KGW/SIR (bias before tempering), MorphMark/SynthID (return log-probs of a
reweighted distribution) and PF (its own decoder) sample from the same base distribution.
The cell starts from `BenchmarkConfig.from_env()`, so every `WMBENCH_*` variable documented there
(`WMBENCH_SMOKE_TEST`, `WMBENCH_MODEL`, `WMBENCH_ORACLE`, `WMBENCH_RUN_NAME`, `WMBENCH_VARIANTS`, `WMBENCH_DATASETS`,
`WMBENCH_DEVICE`, `WMBENCH_DTYPE`, `WMBENCH_SEED`, `WMBENCH_RESULTS`, `WMBENCH_N_SAMPLES`, `WMBENCH_ROBUSTNESS`) works for
headless execution; the constants in the cell override the environment when set.
""")
code("""
# Notebook overrides. Leave a value at None to take it from the WMBENCH_* environment (see
# BenchmarkConfig.from_env) or the preset default; set it here to pin it for interactive use.
SMOKE_TEST      = None      # None -> WMBENCH_SMOKE_TEST env (default: smoke); False for the real run
MODEL_NAME      = None      # <-- PLACEHOLDER lives in BenchmarkConfig: "facebook/opt-1.3b"; e.g. "Qwen/Qwen2.5-1.5B-Instruct"
ORACLE_MODEL    = None      # PPL oracle; "none" -> score with the generator
RUN_NAME        = None
VARIANTS_TO_RUN = None      # e.g. ("none", "KGW", "MorphMark", "PF", "SynthID", "SIR", "SemStamp")
DATASETS_TO_RUN = None      # e.g. ("c4", "wmt16_de_en", "cnn_dailymail", "humaneval")
RUN_ROBUSTNESS  = None      # True -> word-deletion attack on C4
WATERMARK_OVERRIDES = {}    # e.g. {"KGW": {"delta": 1.5, "gamma": 0.25}} - keys as in configs/watermarks/*.json

if SMOKE_TEST is not None:
    os.environ["WMBENCH_SMOKE_TEST"] = "1" if SMOKE_TEST else "0"
os.environ.setdefault("WMBENCH_SMOKE_TEST", "1")   # the notebook defaults to the smoke preset
overrides = {k: v for k, v in {
    "model_name": MODEL_NAME, "variants": VARIANTS_TO_RUN, "datasets": DATASETS_TO_RUN,
    "run_robustness": RUN_ROBUSTNESS, "run_name": RUN_NAME}.items() if v is not None}
if ORACLE_MODEL is not None:
    overrides["oracle_model_name"] = None if str(ORACLE_MODEL).lower() == "none" else ORACLE_MODEL
cfg = BenchmarkConfig.from_env(watermark_overrides=WATERMARK_OVERRIDES, **overrides)
if RUN_NAME is None and not os.environ.get("WMBENCH_RUN_NAME"):
    cfg = cfg.with_updates(run_name="smoke" if cfg.smoke else "nb-" + cfg.model_name.split("/")[-1])
SMOKE_TEST = cfg.smoke
if SMOKE_TEST:
    display(Markdown("> **SMOKE TEST mode** - tiny model and a handful of samples; numbers are not meaningful. "
                     "Set `SMOKE_TEST = False` (or `WMBENCH_SMOKE_TEST=0`) for the real run."))
print(json.dumps(cfg.as_dict(), indent=2))
print("config hash:", cfg.config_hash(), "| results ->", cfg.run_dir)
""", tags=["parameters"])
md("""
The resolved per-variant hyperparameters (from `configs/watermarks/*.json`, plus your overrides) and what each
detector's score means:
""")
code("""
rows = []
for name in cfg.variants:
    spec = VARIANTS[name]
    params = json.loads((CONFIG_DIR / spec.config_file).read_text()) if spec.config_file else {}
    params.update(cfg.watermark_overrides.get(name, {}))
    params.pop("algorithm_name", None)
    rows.append({"variant": name, "family": spec.family, "score": spec.score_name,
                 "shipped threshold": spec.fixed_threshold, "nominal FPR": spec.nominal_fpr,
                 "hyperparameters": ", ".join(f"{k}={v}" for k, v in params.items() if not isinstance(v, (list, dict, str)))[:120],
                 "citation": spec.citation})
pd.set_option("display.max_colwidth", 140)
display(pd.DataFrame(rows))
print("shared sampling:", SAMPLING, "| max_new_tokens:", cfg.max_new_tokens)
""")

# --------------------------------------------------------------------------- model
md("""
## 2. Load the model

The `Runner` owns the model, tokenizer, prompt formatter and result directory. Loading is lazy; the first
access below triggers the download (or the offline cache). `vocab_size` handed to MarkLLM is the **logit
dimension** of the model, not `len(tokenizer)` (OPT: 50272 vs 50265) - green lists and SIR's mapping are
indexed against the logits.
""")
code("""
runner = Runner(cfg, show_progress=True)
model, tok = runner.model, runner.tokenizer
from wmbench.models import logits_vocab_size, max_positions
print(f"device={runner.device} dtype={runner.dtype} | logits={logits_vocab_size(model)} tokenizer={len(tok)} "
      f"| context={max_positions(model)} | chat template={'yes' if runner.formatter.is_chat else 'no'}")
""")

# --------------------------------------------------------------------------- variants
md("""
## 3. The watermarks, one at a time

Two families are represented. **Logit-level schemes** (KGW, MorphMark, SIR) add a `LogitsProcessor` to
`model.generate`; **sampling-level schemes** (SynthID's tournament, Permute-and-Flip's keyed decoder, SemStamp's
rejection sampling) change how the next token or sentence is chosen. Every variant below receives the *same
prompt token ids* and the same sampling settings; the helper generates one C4 continuation, then scores the
watermarked text, an unwatermarked continuation from the same prompt, and the human continuation.

Scores are on different scales (z-score, mean g-value, -log10 p, mean similarity) and are never averaged
across variants; the benchmark compares them through threshold-free metrics (AUROC) and threshold-at-fixed-FPR
metrics instead.
""")
code("""
from wmbench.data import load_task
demo_sample = load_task("c4", 1, runner.templates)[0]
demo_ids, demo_text = runner.formatter.encode(demo_sample.task_prompt), runner.formatter.format(demo_sample.task_prompt)
baseline = build_variant("none", model, tok, runner.device, cfg.max_new_tokens["c4"])
torch.manual_seed(cfg.seed)
none_out = baseline.generate(demo_ids, demo_text)
human_text = demo_sample.human_text
print("PROMPT:", demo_sample.task_prompt[:300])
print("\\nUNWATERMARKED:", none_out.continuation[:400])

def demo(name, max_new_tokens=None):
    v = build_variant(name, model, tok, runner.device, max_new_tokens or cfg.max_new_tokens["c4"],
                      overrides=cfg.watermark_overrides.get(name), semstamp_n_max=cfg.semstamp_n_max)
    torch.manual_seed(cfg.seed)
    t0 = time.time(); out = v.generate(demo_ids, demo_text); dt = time.time() - t0
    print(f"[{name}] {len(out.continuation_ids)} tokens in {dt:.1f}s | score = {v.spec.score_name}, "
          f"shipped threshold = {v.spec.fixed_threshold}")
    print("WATERMARKED:", out.continuation[:400].replace("\\n", " "), "\\n")
    for label, text in (("watermarked", out.continuation), ("unwatermarked", none_out.continuation), ("human", human_text)):
        d = v.detect(text)
        print(f"  {label:13s} score={d.score:8.3f}  flagged={d.is_watermarked}  tokens={d.n_tokens}"
              + (f"  ({d.error})" if d.error else ""))
    return out
""")

md("""
### 3.1 KGW - the reference green-list watermark

At each step the previous token is hashed to seed a permutation of the vocabulary; the first `gamma` fraction is
the *green list* and `delta` is added to those logits before sampling. Detection re-derives the green list for
every position and counts hits: with `T` scored tokens and `g` green ones, `z = (g - gamma T) / sqrt(T gamma (1-gamma))`.
Key hyperparameters: `gamma=0.5`, `delta=2.0`, `prefix_length=1`, `z_threshold=4` (one-sided normal FPR about 3e-5).
Strong and cheap; the bias is the same at every step, so low-entropy text (code, lists) pays the largest quality price.
""")
code("""_ = demo("KGW")""")

md("""
### 3.2 MorphMark - adaptive green-list watermark

MorphMark keeps KGW's green list but replaces the constant logit bias with a probability shift that depends on the
mass `P_G` the green list already has: below `p_0` nothing is changed, above it a fraction `r(P_G)` of the remaining
mass is moved to green tokens (`type="exp"`: `r = exp(k_exp P_G) - 1`). Positions where the model is confident about
a red token are left alone, which is where fixed-bias schemes hurt quality most. Detection is KGW's z-score; the
shipped threshold is `z=2` (nominal FPR about 2.3%), so compare it at equal FPR, not at its own threshold.
""")
code("""_ = demo("MorphMark")""")

md("""
### 3.3 PF - Permute-and-Flip decoder

Permute-and-Flip is a decoding rule: pick `argmax_v [log p(v) - log r(v)]` where `r` is a pseudo-random vector
seeded by the previous `ngram=8` tokens and a secret key. Without the key this is equivalent to sampling from `p`
(it is a Gumbel-style trick), so the output distribution is unbiased; with the key the emitted tokens have
systematically small `r`. Detection sums `-log r(token)` over the text: under no watermark the sum of `k` such
terms is `Gamma(k, 1)`. **wmbench reports `-log10` of that Gamma tail p-value** (p < 0.01 = flagged) instead of
MarkLLM's raw sum, which grows with length and whose threshold subtracts the n-gram size twice.
The decoder is a custom loop (no HF `generate`), so wmbench feeds it the shared prompt ids and stops at EOS.
""")
code("""_ = demo("PF")""")

md("""
### 3.4 SynthID-Text - tournament sampling

Google's SynthID-Text draws `ngram_len=5`-gram-keyed pseudo-random *g-values* (0/1) for every candidate token under
30 layered keys and runs a tournament: at each layer the probability mass is tilted toward tokens with g = 1.
In the default *non-distortionary* mode a context that already appeared is not watermarked again, which keeps the
text distribution unchanged in expectation. Detection is the mean g-value over the text's n-grams (0.5 expected
without watermark; shipped threshold 0.52). Two MarkLLM details are handled here: the processor's context history
is reset before every prompt (MarkLLM carries it across prompts, under-watermarking later samples), and its
internal temperature is pinned to the shared sampling temperature.
""")
code("""_ = demo("SynthID")""")

md("""
### 3.5 SIR - semantic invariant robust watermark

SIR makes the logit bias a function of *meaning*: the preceding text (in `chunk_length=10`-word chunks) is embedded
with a compositional-BERT sentence encoder, a small trained transform network maps the embedding to a 300-d
vector, and each vocabulary token is tied to one of those dimensions through a fixed mapping; the token's bias is
`delta` times the (sign-scaled) value of its dimension. Paraphrases with the same meaning therefore produce the same
biases, which is what makes the watermark robust to rewriting. Detection averages the bias signs of the observed
tokens (`z_threshold=0.2` on a mean in [-1, 1]). It is the slowest logit-level scheme (one BERT forward per
generated token) and needs the embedder + transform weights (`artifacts/sir/`).
""")
code("""_ = demo("SIR")""")

md("""
### 3.6 SemStamp (optional) - sentence-level semantic watermark

SemStamp generates one sentence at a time, embeds it (SBERT), hashes the embedding with random hyperplanes into
`2^dimension_d = 8` buckets, and only accepts the sentence if its bucket is in the `gamma=25%` "valid" set seeded by the
previous sentence (otherwise it re-samples, up to `N_max` times). Detection counts valid sentences and reports a z-score.
It is robust to paraphrasing but costs many generations per sentence and only measures on C4 here. MarkLLM's
implementation is OPT-only, hard-codes temperature 0.7 and suppresses EOS; wmbench uses the shared prompt ids and
sampling settings, keeps EOS, bounds the continuation by the shared token budget and accepts the last candidate after
`N_max` rejections. Enable it by adding `"SemStamp"` to `VARIANTS_TO_RUN`.
""")
code("""
if "SemStamp" in cfg.variants:
    _ = demo("SemStamp")
else:
    print("SemStamp not selected (add it to VARIANTS_TO_RUN to run it; expect minutes per sample on CPU).")
""")

# --------------------------------------------------------------------------- stages
md("""
## 4. Generate + detect (stage `generate`)

For each dataset and variant: format the task prompt (few-shot `German: ... English:` for base models, an
instruction for chat models; `TL;DR:` for summaries), generate with a per-sample seed derived from
`(seed, dataset, sample_id, variant)`, time it, and append a record to `results/<run>/generations/<dataset>/<variant>.jsonl`.
On C4 every watermark variant then scores its own continuations and the (equally truncated) human continuations.
Records are cached: interrupt and re-run at will. On a cluster this stage is what the SLURM array parallelises.
""")
code("""runner.stage_generate()""")

md("""
## 5. Detection against unwatermarked generations (stage `detect_none`)

Human text is the primary negative class (that is what a deployed detector sees). The `none` variant's continuations
are a second, harder negative set: same model, same prompts, same length, just no watermark. This stage needs the
`none` generations from stage 4.
""")
code("""runner.stage_detect_none()""")

md("""
## 6. Quality (stage `quality`)

* **Perplexity** of each C4 continuation *conditioned on its prompt* under the oracle model (the generator is freed
  first when the oracle differs). The report shows the token-weighted corpus PPL and the median per-sample PPL, plus a
  `human` row scored the same way.
* **Diversity**: repeated 2/3/4-gram rates and their product `diversity = prod(1 - rep_n)`; MarkLLM's `log_diversity`
  formula is reported too for comparability.
* **BLEU** (sacreBLEU, corpus level) on WMT16 de->en, **ROUGE-1/2/L** on CNN/DailyMail, **pass@1** on HumanEval if enabled
  (model code is executed in a subprocess with a timeout).
""")
code("""
# Drop every notebook reference to the generator so it can really be freed before the oracle is loaded.
for _name in ("model", "tok", "baseline", "_"):
    globals().pop(_name, None)
runner.stage_quality()
""")

# --------------------------------------------------------------------------- results
md("""
## 7. Results

`stage_report` is pure pandas over the JSONL files, so it can be re-run any time (also for runs produced by the CLI:
construct a `BenchmarkConfig` with the same `run_name` and call `wmbench.report.build_report(cfg)`).
""")
code("""
out = runner.stage_report()
if cfg.smoke:
    display(Markdown("> **SMOKE TEST** - pipeline validation only; numbers are not meaningful."))
display(Markdown("### Main table (one row per variant)"))
display(out["main"].round(3))
""")
code("""
display(Markdown("### Detection (C4): negatives = human continuations (`_human`) or `none` generations (`_none`)"))
display(out["detection"].round(3))
""")
code("""
display(Markdown("### Quality per dataset"))
display(out["quality"].round(3))
display(Markdown("### Paired differences vs `none` on the same prompts (bootstrap 95% CI)"))
display(out["paired_deltas"].round(3))
""")
code("""
for p in out["plots"]:
    display(Markdown(f"**{p.stem}**")); display(Image(filename=str(p)))
print("Full report:", cfg.run_dir / "summary.md")
""")

md("""
## 8. How to read this

* **TPR @ 1% FPR vs human** is the headline detectability number: the fraction of watermarked continuations flagged
  when the threshold is set so that at most 1% of human texts are flagged. It needs >= 100 negatives to mean anything
  (`fpr_warning_*` in `detection_summary.csv` flags smaller sets); the smoke preset cannot deliver it.
* **AUROC** is threshold-free and comparable across score scales.
* `fixed_thr_tpr/fpr` use each algorithm's *shipped* threshold. MorphMark's `z=2` and SIR's `0.2` are far looser than
  KGW's `z=4`, so those columns reward loose thresholds; use them to judge the shipped defaults, not the schemes.
* **PPL** is relative to `none` and to the `human` row; the paired deltas in section 7 remove prompt-to-prompt variance.
  Distortion-free schemes (PF, SynthID non-distortionary) should sit near `none`; bias schemes (KGW, SIR) pay in PPL
  and gain in detectability - the classic trade-off (see `plots/ppl_vs_tpr.png`).
* **BLEU / ROUGE** with a *base* placeholder model are low for every variant; the differences to `none` (paired deltas)
  are still informative. An instruct model gives more meaningful absolute values.
* Scores are length-dependent; all three text classes are truncated to the same token budget and samples whose
  human continuation is shorter than `min_detect_tokens` are excluded for every variant.
* Log-likelihood benchmarks (MMLU-style multiple choice) are **invariant** to decoding-time watermarks and are
  therefore not part of this comparison.
""")

md("""
## 9. Customising

* **Your model**: set `MODEL_NAME` (and a larger `ORACLE_MODEL` of the same family for PPL; `"none"` to score with the
  generator). SIR needs a token mapping for the model's logit dimension; one is generated deterministically on first
  use (`artifacts/sir/300_mapping_<vocab>_seed0.json`) when MarkLLM does not ship one.
* **Hyperparameters**: `WATERMARK_OVERRIDES = {"KGW": {"delta": 1.0}}` etc.; keys are those of `configs/watermarks/*.json`.
  A sweep is a loop over run names with different overrides.
* **Another MarkLLM algorithm**: add a `VariantSpec` in `wmbench/watermarks.py` (algorithm key, config file,
  score name, threshold) and copy its JSON from `third_party/MarkLLM/config/`. Logit-processor algorithms work out of
  the box; algorithms with their own decoding loop need a small adapter like the PF one.
* **Cluster**: `scripts/prepare_assets.py` on the login node, then `scripts/slurm/array_generate.sbatch` +
  `scripts/slurm/merge.sbatch`. See README.
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
