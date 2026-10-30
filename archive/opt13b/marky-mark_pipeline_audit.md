# marky-mark: pipeline audit for the paper (run `archive/opt13b`, formerly `run_results/opt13b`)

The numbers in the results section match what the code produces. To check, they were rebuilt from the
per-sample records with `scripts/paper_assets.py`; the script's built-in cross-checks against `summary.md`
all passed. Nothing in the repo was changed.

## Mismatches to fix in the paper first

- **Word-deletion TPR for SynthID and SIR is really measured at about 3% FPR, not 1%.** These two scores
  are averages per token. Attacked texts are shorter (about 123 tokens vs 181 for the clean human texts),
  so their scores spread more, but the threshold was set on the clean human texts. `compute_robustness`
  estimates the real FPR at 3.1% (SynthID) and 2.9% (SIR). For DIPPER it is 1.3%.
- **The detection confidence intervals do not resample prompts.** The captions say "bootstrap CIs over
  prompts", but `metrics.detection_bootstrap` (`wmbench/metrics.py:344`) resamples watermarked and human
  scores independently. Only the method-vs-method comparisons under attack resample by prompt.
- **The "d′" is not textbook d′.** It is the median over watermarked texts of (s − μ_H)/σ_H, not a
  difference of means. There is no interval on it.
- **There are two different perplexity-change numbers.** `paired_deltas.csv` and `degradation.csv` use the
  geometric mean of per-sample PPL (B=1000): KGW +37.0%. The paper tables use the ratio of corpus PPL
  (B=2000): KGW +38.0%. Cite one consistently.
- **The GPU model is not stored in the records.** The captions hard-code "RTX A4000". All generations came
  from one job (2326862 on tcml-node18), which fits `all_in_one.sbatch` (it requests an A4000). The attack
  and attacked-text detection stages ran later on tcml-node3 (the L40S script).
- **SIR on CNN/DailyMail:** for 68 of 100 prompts, the context passed to SIR's embedder was over its
  512-token limit.
- **AUROC "1.000" is rounded.** Only SynthID is exactly 1.0. MorphMark is 0.9997 (CI lower bound 0.999),
  KGW 0.99994, PF 0.99999.

## Detection and scoring

### Detection score per method

All scores come from MarkLLM at commit `0a4fe8c` and are wrapped in `WatermarkVariant.detect`
(`wmbench/watermarks.py:270`).

- **KGW and MorphMark:** one-proportion z-score of green tokens over all tokens after the first. Repeated
  tokens are not filtered out.
- **PF:** wmbench's own score: −log10 of the Gamma(k,1) upper-tail p-value of the summed −log r, where k is
  the number of scored tokens (`_detect_pf`, `wmbench/watermarks.py:290`). Every 8-gram is scored, repeats
  included.
- **SynthID:** mean g-value, with repeated contexts and EOS masked out (non-distortionary mode).
- **SIR:** mean of the negative similarity over tokens. A sentence is only scored once at least 10 words of
  context precede it.
- **Same quantity for AUROC, TPR and d′?** Yes for AUROC and TPR, and for d′ except PF. There d′ uses the
  normal approximation (S − k)/√k (`linear_scores`, `scripts/paper_assets.py:216`).

### Negative classes

- Human continuations and unwatermarked (`none`) generations are evaluated separately (`_human` and
  `_none` columns in `detection_summary.csv`).
- The paper's headline numbers use human texts only.
- Only three statistics pool both classes: the zero-false-positive TPR, the separation gap, and the
  calibration figure.

### d′ (median margin)

- μ_H and σ_H are computed per detector over all of that detector's kept human samples, not per prompt.
  σ uses ddof=1 (`standardized_scores`, `scripts/paper_assets.py:232`).
- The reported value is the median over watermarked texts. The 1st percentile is also stored.
- No clipping and no epsilon.
- Under attack, the same clean-human μ and σ are reused.
- Recomputed medians: PF 27.22, SynthID 15.71, KGW 8.01, MorphMark 7.29, SIR 5.41. These match 27.2, 15.7,
  8.0, 7.3 and 5.4.

### TPR at a fixed FPR

- The threshold is the (⌊FPR·n⌋+1)-th highest negative score, and a text counts as detected only if its
  score is strictly above it (`metrics.tpr_at_fpr`, `wmbench/metrics.py:226`).
- With n=460 this is the 5th-highest human score, so the actual FPR is 0.87%.
- Both 1% and 5% are computed.
- TPR@1%: KGW 99.8, MorphMark 99.8, PF 100, SynthID 100, SIR 98.0.

### Shipped thresholds

- KGW (z > 4), MorphMark (z > 2), SynthID (0.52) and SIR (0.2) use MarkLLM's own `is_watermarked`, which
  reads the threshold from the config JSON with a strict ">". This was checked against every scored row.
- PF uses p < 0.01, defined in wmbench. MarkLLM's PF threshold is not used.
- `VariantSpec.fixed_threshold` repeats these values by hand for the tables. Editing a JSON would not
  update it.
- What is reported: TPR on watermarked text (so FNR = 1 − TPR), FPR on human text, FPR on unwatermarked
  text, and TPR at the shipped threshold after each attack.

| Method | TPR | FPR human | FPR unwatermarked | Nominal FPR |
|---|---|---|---|---|
| KGW | 98.9 | 0.0 | 0.0 | 0.003% |
| MorphMark | 99.8 | 2.8 | 2.2 | 2.3% |
| PF | 100 | 0.7 | 1.8 | 1.0% |
| SynthID | 100 | 0.4 | 0.6 | none defined |
| SIR | 98.0 | 0.9 | 0.9 | none defined |

### Bootstrap

- All intervals are percentile intervals with seed 0. The paper uses B=2000; the run's `summary.md` uses
  B=500 for detection and B=1000 for quality, so its intervals differ slightly from the paper's.
- **Detection:** not paired (see the mismatch above).
- **Quality:** paired by prompt, against `none`.
- **Robustness:** method-vs-method differences in TPR and AUROC are paired by prompt.
- **d′:** no interval.

### Exclusions (why n is 446–462)

- **Common budget:** T = min(200, watermarked length, human length, `none` length), counted in OPT tokens
  without special tokens (`common_token_budget`, `wmbench/evaluate.py:89`).
- **Minimum length:** a sample is dropped if T < 50 (`min_detect_tokens`, applied in
  `report.kept_sample_ids`).
- **What caused the drops:** in every case the watermarked text or the `none` text hit EOS before 50
  tokens. Human text was never the short one.
- **Counts dropped:** KGW 40, MorphMark 41, PF 43 + 1 empty output, SynthID 37 + 1 empty output, SIR 54.
- **Detector failures:** caught in a try/except and recorded as NaN with the error message. All of them
  are on texts of 17 tokens or fewer, which were already excluded. Kept samples have 0 NaNs.

## Generation settings

- **Sampling:** do_sample, temperature 1, top_p 1, top_k 0, repetition penalty 1, no n-gram blocking
  (`wmbench/config.py:25`). The model's Hub generation config is discarded.
- **Batch size:** 1.
- **Max new tokens:** C4 200, WMT16 64, CNN/DailyMail 96.
- **"Temperature 1, no truncation" holds for generation.** Note that PF replaces sampling with its own
  keyed decoder.
- **Seeds:** base seed 0. The per-sample seed is a hash of (seed, dataset, sample, variant), so it differs
  between variants. Watermarked and `none` texts do not share a random stream, though they do share
  prompts.
- **Hyperparameters:** all five configs are identical to MarkLLM's defaults at the pinned commit (diffed).
  KGW and MorphMark: γ 0.5, 1-token context. KGW δ 2. MorphMark: exp morphing, k 1.3, p₀ 0.15. PF: 8-gram
  context. SynthID: 5-gram context, 30 layers, mean detector. SIR: δ 1, chunk length 10.
- **Runtime changes:** PF's `max_seq_len` is raised to 2048 (a buffer size only), and SynthID's temperature
  and top_k are pinned to 1 and 0.

### Prompt format

- **C4:** the first 500 prompts from MarkLLM's `processed_c4.json`, used as-is. Every prompt is 30 tokens
  including BOS (11–29 words); the code does not cut them. The human continuation is the file's
  `natural_text`.
- **WMT16:** 5 example pairs, taken from the last 5 rows of the file, as `German: …\nEnglish: …`. Then
  `German: <src>\nEnglish:`. The first 200 rows are evaluated. The answer is the first non-empty line, cut
  at "German:".
- **CNN/DailyMail:** the first 100 articles, cut to their first 400 words (54 were cut), then
  `\n\nTL;DR:`. The summary is the first paragraph of the output.

## Robustness

- **Word deletion:** each whitespace-separated word is dropped with probability 0.3, seeded per sample.
  Line breaks are lost when the words are rejoined.
- **DIPPER:** lexical diversity 60 (control code 40), order diversity 0, one sentence at a time (split with
  nltk `sent_tokenize`), conditioned on the C4 prompt plus the paraphrase so far. Sampling uses top_p 0.75,
  up to 512 new tokens, bf16.
- **What gets attacked:** only the watermarked text, already cut to T. The output is capped at T tokens.
  Negatives stay clean.
- **After the attack:** detection is fully re-run. The pipeline reports TPR at the clean-human 1%
  threshold, AUROC against clean human text, TPR at the shipped threshold, and attack-strength measures
  (BLEU against the source, embedding similarity, length ratio).
- **Retained margin:** median d′ after the attack divided by clean median d′.
  - Word deletion: KGW 0.61, MorphMark 0.59, PF 0.07, SynthID 0.29, SIR 0.98.
  - DIPPER: KGW 0.26, MorphMark 0.27, PF 0.02, SynthID 0.06, SIR 0.43.
- **"Implied survival":** retained ÷ √(length ratio) for the sum-type scores (KGW, MorphMark, PF); retained
  as-is for SynthID and SIR.
- **Sample counts:** the same as the clean evaluation; no extra filtering. Some attacked texts are as short
  as 24 tokens, since there is no minimum length for them.

## Quality and efficiency

- **Perplexity:** under OPT-2.7B in fp16, conditioned on the prompt (with BOS). Only the continuation
  tokens are scored. The full continuation (up to 200 tokens) is used, not the detection budget. Human text
  is cut to 200 tokens.
- **Diversity:** whitespace tokens, rep_n for n = 2, 3, 4, diversity = ∏(1 − rep_n), averaged over samples.
  This is the SimCTG formulation.
- **BLEU:** sacreBLEU with default settings: 13a tokenizer, exp smoothing, case-sensitive, one reference.
  The sacreBLEU version is not recorded anywhere.
- **ROUGE:** `rouge_score` rouge1, rouge2 and rougeL (not rougeLsum), F-measure, with stemming.
- **Throughput:**
  - One untimed warm-up generation per method and dataset, then one timed run per sample. There are no
    repeats.
  - Detection time is not included.
  - tok/s is total tokens ÷ total seconds.

## Output files

- The committed results are in `run_results/opt13b/`: `summary.md`, `detection_summary.csv` and
  `results.json`.
- To regenerate `numbers.json`, `numbers_robustness.json`, the tables and the figures, run
  `python scripts/paper_assets.py --run run_results/opt13b --out paper`.
