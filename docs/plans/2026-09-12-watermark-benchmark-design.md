# Design: LLM watermark benchmark on top of MarkLLM

Date: 2026-09-12
Status: draft, written autonomously (no user Q&A available); assumptions are listed at the end.

## 1. Goal

Given a Hugging Face causal language model (a placeholder for now), wrap it with several
watermarking schemes from the MarkLLM toolkit and compare the resulting "watermarked models" on
text quality (perplexity, BLEU, ...), detectability, and at least one standard downstream
benchmark. Deliver the comparison as a Jupyter notebook backed by a small, tested Python package,
plus a README.

## 2. Ground truth gathered before designing

Facts verified by reading and running the MarkLLM source (commit `0a4fe8c6`, 2026-09-05):

| Fact | Consequence |
| --- | --- |
| The PyPI wheel `markllm==0.1.5` only ships KGW, Unigram, SWEET, UPV, EWD, SIR, XSIR, EXP family. No MorphMark, PF, SynthID, SemStamp. | Must install MarkLLM from GitHub at a pinned commit, not from PyPI. |
| MarkLLM modules use absolute imports (`from utils.transformers_config import ...`) and default config paths relative to the MarkLLM root. | The MarkLLM root must be on `sys.path`; we pass absolute config paths and never rely on `cwd`. |
| `AutoWatermark.load(name, algorithm_config=path, transformers_config=tc, **overrides)` merges `overrides` into the JSON config. | We can keep MarkLLM's JSON defaults and override per variant from Python. |
| KGW, MorphMark, PF, SynthID, SIR, SemStamp all run on transformers 5.3 / torch 2.10 on CPU and (except SIR/SemStamp, untested) on Apple MPS. | No version pin below 5 is required; we pin a lower bound only. |
| SIR needs `transform_model_cbert.pth` (HF `Generative-Watermark-Toolkits/MarkLLM-sir`) and the embedder `perceptiveshawty/compositional-bert-large-uncased` (1.3 GB). The `.pth` is CUDA-pinned; `torch.load` fails on CPU-only hosts. | Bootstrap downloads both, re-saves the `.pth` with `map_location="cpu"`, and writes a resolved SIR config. |
| SIR's token mapping file is vocab-size specific; for unknown vocab sizes MarkLLM creates an unseeded random mapping. | We generate the mapping ourselves with a fixed seed and persist it under `artifacts/`. |
| SynthID reads `temperature`/`top_k` as attributes of `TransformersConfig`, not from `gen_kwargs`, and applies temperature inside its logits processor. HF then applies the sampling temperature again. | The SynthID builder sets those attributes to the shared sampling values and overrides HF's temperature to 1.0 for that variant only. |
| PF uses its own decoding loop (`temperature`, `top_p` from `gen_kwargs`, no top-k, no `no_repeat_ngram_size`, no `min_new_tokens`). | We use pure multinomial sampling (`top_k=0`, `top_p=1.0`) for all variants so PF and HF-based variants sample from the same base distribution. |
| SemStamp is rejection sampling (up to `N_max` regenerations per sentence) and hard-codes `"opt" in model name`. | Optional variant, off by default, with a thin subclass that lifts the OPT check. |
| MarkLLM's evaluation modules import `openai` and `translate` at module level. | We implement metrics natively (sacrebleu, rouge_score, torch) and only import MarkLLM's watermark classes, dataset loaders and `DynamicThresholdSuccessRateCalculator`. |
| MarkLLM's PPL is computed on the continuation alone, without the prompt as context. | We compute prompt-conditioned continuation perplexity with an oracle model, which is the KGW-paper convention, and document the difference. |
| Datasets (C4 subset, WMT16 de-en, CNN/DailyMail test, HumanEval) ship inside the MarkLLM repo. | No separate dataset download step. |

## 3. Scope

In scope:

- Variants (the "versions" of the model): `none` (baseline sampling), `KGW` (reference), `MorphMark`,
  `PF` (Permute-and-Flip), `SynthID`, `SIR` (semantic invariant robust), optional `SemStamp`.
- Quality metrics: prompt-conditioned perplexity (oracle LM), log diversity, BLEU (WMT16 de-en),
  ROUGE-1/2/L (CNN/DailyMail), optional pass@1 (HumanEval).
- Detection metrics: score distributions on watermarked vs. human vs. unwatermarked-generated
  text; TPR at 1% FPR and best-F1 (MarkLLM's dynamic threshold calculator); AUROC; accuracy at each
  algorithm's own fixed threshold.
- Efficiency: seconds and tokens per second per generation.
- Optional robustness: word-deletion attack, TPR after attack.
- Resumable result caching, summary tables, plots, README, tests.

Out of scope (YAGNI): vLLM integration, paraphrase attacks that need OpenAI or DIPPER, MarkLLM
visualisers, training-based variants (UPV, TS, Adaptive), seq2seq translation models.

## 4. Architecture

```
marky-mark/
  README.md
  requirements.txt              # runtime deps (torch/transformers lower bounds, sacrebleu, ...)
  pyproject.toml                # `pip install -e .` for the wmbench package
  .gitignore                    # third_party/, artifacts/, results/, .venv/
  configs/watermarks/*.json     # per-algorithm configs (copies of MarkLLM defaults, editable)
  wmbench/
    __init__.py
    bootstrap.py                # ensure_markllm() clone+pin+sys.path; ensure_sir_assets(); nltk data
    config.py                   # BenchmarkConfig dataclass, SMOKE/FULL presets, GenerationConfig
    models.py                   # device/dtype selection, load_causal_lm, PromptFormatter (chat templates)
    watermarks.py               # registry: build_variant(name, ...) -> WatermarkVariant; NoWatermark
    data.py                     # dataset loading via MarkLLM loaders; task prompt templates
    generation.py               # generate_records(): cached, timed generation per (variant, dataset)
    metrics.py                  # ppl, log_diversity, bleu, rouge, pass@1, detection_metrics, auroc
    attacks.py                  # word_deletion (optional robustness)
    evaluate.py                 # orchestration: run_benchmark(config) -> results dir
    report.py                   # summary DataFrame, markdown table, plots
  notebooks/watermark_benchmark.ipynb
  scripts/run_benchmark.py      # CLI equivalent of the notebook
  tests/                        # pytest; fast unit tests + slow integration tests
  docs/plans/                   # this document
```

Data flow:

1. `ensure_markllm()` clones MarkLLM at the pinned commit into `third_party/MarkLLM` and inserts
   it into `sys.path`. `ensure_sir_assets()` downloads the SIR transform model and embedder into
   `artifacts/sir/`, re-saves the checkpoint on CPU, and returns a resolved config path.
2. `load_causal_lm()` loads the placeholder model and tokenizer once. `TransformersConfig` is built
   from `GenerationConfig` with explicit `temperature`, `top_p=1.0`, `top_k=0`, `max_new_tokens`,
   `min_new_tokens`, `no_repeat_ngram_size`. `vocab_size` is the model's logit dimension, not
   `len(tokenizer)`.
3. `build_variant()` returns a `WatermarkVariant` with `generate(prompt) -> str`,
   `detect(text) -> (score, is_watermarked)` and `name`. It wraps MarkLLM instances, handles the
   SynthID temperature fix, SIR asset resolution, and the SemStamp subclass. `none` uses plain
   `model.generate` with the same `gen_kwargs`.
4. `generate_records()` iterates a dataset, formats the task prompt (translation, summarisation,
   completion), generates, extracts the continuation, times it, and appends a JSON record to
   `results/<run>/<dataset>/<variant>.jsonl`. Re-runs skip already-present sample ids.
5. `metrics.py` scores records: quality metrics per record; detection metrics per variant
   comparing watermarked continuations against human text (natural continuation or reference) and
   against `none` continuations.
6. `report.py` aggregates into one table per metric family, writes `summary.csv`,
   `summary.md`, and PNG plots.

The notebook calls the same functions in order, with explanatory markdown, one demo cell per
watermark showing a sample generation and its detection score, then the evaluation sections, then
the summary. A `SMOKE_TEST` switch selects `facebook/opt-125m`, 8 samples, 64 new tokens so the
whole notebook runs on CPU in minutes; the full preset targets a single GPU.

## 5. Key decisions

- **Placeholder model**: `facebook/opt-1.3b`. It is MarkLLM's reference model, non-gated, and the
  SIR mapping for its vocabulary ships with MarkLLM. Any HF causal LM can be substituted; chat
  models are auto-formatted with their chat template.
- **Oracle for perplexity**: `facebook/opt-2.7b` by default (a larger model of the same family, as
  in the KGW and MarkLLM papers). When `oracle_model_name` is `None`, the generator scores itself.
- **Standard benchmarks**: WMT16 de-en (BLEU) and CNN/DailyMail (ROUGE) run by default via prompted
  generation with the single placeholder model; HumanEval pass@1 is available behind a flag because
  it executes model-written code (in a subprocess with a timeout) and only makes sense for
  code-capable models. C4 provides the PPL/detection prompts as in the MarkLLM paper.
- **Fair sampling**: one shared sampling configuration across variants; documented exceptions
  where MarkLLM's implementation forces them (PF decoding loop, SynthID internal temperature).
- **Vendoring**: MarkLLM is cloned at runtime at a pinned commit (`wmbench/bootstrap.py`), not
  copied into the repo, so licensing and updates stay clean and the repo stays small.
- **Metrics implemented natively** rather than through MarkLLM pipelines to avoid the OpenAI import
  chain and to get per-sample records, timing, and resumability.

## 5a. Cluster execution (primary target)

The user runs this on a cluster, not locally. Consequences:

- `scripts/run_benchmark.py` is a first-class entry point, not a convenience: it accepts a preset
  and CLI overrides (`--model`, `--variants`, `--datasets`, `--n-samples`, `--run-name`,
  `--device`, `--dtype`) and writes everything under `results/<run-name>/`.
- Variants are independent units of work. `--variants SynthID` runs one variant; a SLURM array
  job can fan out one variant per task, and `scripts/run_benchmark.py --report-only` (or the
  notebook's summary section) merges the per-variant JSONL files afterwards.
- Compute nodes are often offline. `scripts/prepare_assets.py` runs on a login node and prefetches
  the MarkLLM clone, SIR weights and embedder, NLTK tokenizer data, and the generator and oracle
  models into the Hugging Face cache (`HF_HOME`). Bootstrap functions are no-ops when assets exist,
  and the code respects `HF_HUB_OFFLINE=1`.
- Example SLURM scripts live in `scripts/slurm/` (single job and array-over-variants).
- Python >= 3.10 is required because MarkLLM uses `X | Y` union annotations at import time.
- The notebook remains the narrative deliverable and can be executed headlessly with
  `jupyter nbconvert --execute` on a GPU node; both paths call the same `wmbench` functions.

## 6. Error handling

- Missing GPU: device auto-selects `cuda`, then `mps`, then `cpu`; dtype defaults to fp16 on CUDA
  and fp32 elsewhere. A warning explains expected runtime.
- Missing MarkLLM or SIR assets: bootstrap functions are idempotent and raise clear errors
  pointing at the exact download that failed.
- A variant failing on one sample logs the error into the record (`error` field) and continues;
  metrics skip errored records and report the count.
- Interrupted runs: JSONL caching makes every stage resumable.
- Detection on short texts: SIR and SemStamp return NaN when fewer than one chunk/sentence is
  available; NaNs are excluded and counted.

## 7. Testing

- Unit tests (CPU, seconds): continuation extraction, prompt formatting, log diversity, BLEU and
  ROUGE wrappers, AUROC and TPR@FPR against hand-computed values, JSONL cache resume logic, config
  presets.
- Integration tests (marked `slow`, need network): bootstrap clone, each variant on
  `facebook/opt-125m` for one prompt, PPL with the generator as oracle.
- Notebook check: execute the notebook in smoke mode with `nbclient` before committing.

## 8. Assumptions to confirm with the user

1. The comparison target is one causal LM wrapped with different watermarks, not different models.
2. "Semantic" refers to SIR (Semantic Invariant Robust watermark, ICLR 2024); SemStamp is offered
   as an optional second semantic variant.
3. Standard benchmarks: WMT16 de-en and CNN/DailyMail by default, HumanEval optional.
4. A single-GPU cluster node is the intended full-run environment (confirmed by the user mid-task);
   the local 8 GB Apple M1 is for smoke tests only.

## 9. Review outcomes (2026-09-12, four-lens design review)

An automated review panel (MarkLLM API fidelity, benchmark methodology, cluster engineering, notebook and
README usability) produced 27 ranked changes. Adopted, with the module that implements each:

| id | change | where |
|---|---|---|
| C1 | Reset SynthID's logits-processor state before every generation (MarkLLM carries context history across prompts); pin `temperature`/`top_k` attributes on `TransformersConfig` | `watermarks.py` |
| C2 | One shared sampling configuration, fixed: `do_sample, temperature=1, top_p=1, top_k=0`; no `min_new_tokens`, no `no_repeat_ngram_size` | `config.SAMPLING` |
| C3 | Token-level generation: every variant consumes the same prompt ids, continuation = sliced ids, detectors see continuation only | `watermarks.py`, `generation.py` |
| C4 | PF subclass taking prompt ids, fp32 logits, EOS stop, `max_seq_len` = model context | `watermarks._pf_generate_from_ids` |
| C5 | PF score = `-log10` Gamma-tail p-value with k = scored tokens; decision p < 0.01 | `watermarks._detect_pf` |
| C6 | Prompt-length policy: article word cap, explicit context check (error record, never silent truncation), `prompt_tokens`/`prompt_truncated` recorded | `data.py`, `watermarks.generate` |
| C7 | Detection on C4 only; equal token budget for watermarked / human / none; common min-length filter; detector failures -> NaN + counts | `evaluate.py`, `report.py` |
| C8 | FULL = 500 C4 prompts; native AUROC / TPR@FPR / best-F1 with bootstrap CIs; warning when negatives < 1/FPR; smoke banner | `metrics.py`, `report.py` |
| C9 | Per-task `max_new_tokens`, 5-shot translation prompts from non-evaluated rows, `TL;DR:` summaries, post-hoc answer trimming, corpus BLEU + paired deltas; SemStamp restricted to C4 | `data.py`, `report.py` |
| C10 | Detection in the same job/device type as generation; device recorded per record; report is pure pandas | `evaluate.py` |
| C11 | Stages `generate` / `detect_none` / `quality` / `report`; SLURM array over variants + merge job | `evaluate.py`, `scripts/slurm/` |
| C12 | SIR assets resolved to absolute paths and validated (mapping length = logit dim) before MarkLLM sees them | `watermarks.py`, `bootstrap.py` |
| C13 | SemStamp subclass with shared sampling, any model, N_max fallback; warm-up generation before timing | `watermarks.py`, `generation.py` |
| C14 | Per-sample seed `hash(seed, dataset, sample_id, variant)`; attack seeds likewise | `generation.py`, `evaluate.py` |
| C15 | Results layout, record schema, `manifest.json` with config hash; resume refuses hash mismatch; fsync per record | `generation.py`, `evaluate.py` |
| C16 | OOM retry once, CUDA errors re-raised, abort after 5 consecutive failures, non-zero exit on zero successes | `generation.py` |
| C17 | Diversity reported as `rep_n` + SimCTG `diversity`; MarkLLM's formula kept alongside for comparability (its origin could not be confirmed either way) | `metrics.diversity_metrics` |
| C18 | Corpus (token-weighted) PPL + median, human anchor row, oracle explicit | `metrics.py`, `evaluate.py` |
| C19 | Fixed-threshold TPR and FPR (not accuracy); `VariantSpec` carries score name, threshold, nominal FPR | `watermarks.py`, `report.py` |
| C20 | Main table, paired deltas vs none, score histograms, ROC overlay, PPL-vs-TPR scatter, task bars | `report.py` |
| C21 | Notebook `parameters` cell with `WMBENCH_*` env overrides; nbconvert command with no timeout | `scripts/build_notebook.py`, `scripts/slurm/notebook.sbatch` |
| C22 | `prepare_assets.py`, pinned checkout verification, dataset path assertions | `scripts/prepare_assets.py`, `bootstrap.py` |
| C23 | Offline env vars in SLURM templates, HF revisions written to `artifacts/assets.json` | `scripts/` |
| C24 | CLI flags for stage, variants, datasets, results root, device, seed, resume; per-stage log files | `cli.py` |
| C25 | Word-deletion robustness scored at the threshold fixed on clean human negatives | `attacks.py`, `report.py` |
| C26 | Prompt formatter emits token ids (chat template tokenized directly) | `models.PromptFormatter` |
| C27 | README and notebook content per the review's outline | `README.md`, notebook |

Not adopted: a GSM8K exact-match track (unconfirmed need; near-zero for the placeholder), back-translation
attacks, multiple generation seeds per prompt, batched generation (PF and SemStamp cannot batch), vendoring
MarkLLM, YAML as a third configuration surface, a non-1 shared temperature, and a lock file from a cluster run
(cannot be produced locally; generate `pip freeze > requirements-lock.txt` after the first successful cluster run).
