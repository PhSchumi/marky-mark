# Results summary: `qwen15b-20261019-173051`

This run benchmarks five text watermarking methods (KGW, MorphMark, PF, SynthID, SIR) on `Qwen/Qwen2.5-1.5B-Instruct`, with the same setup as the `opt13b` run (`facebook/opt-1.3b`). It adds GSM8K (math word problems) as a fourth task. SemStamp is not part of this run.

The figures for this run are in the Qwen section of `../plots_tables.ipynb`.

## Before comparing
- **Different perplexity judge:** Qwen's perplexity is judged by `Qwen2.5-7B-Instruct`, OPT's by `opt-2.7b`. Only the change vs. unwatermarked text can be compared, not the raw values.
- **Shorter texts:** Qwen stops writing earlier, so detection is scored on about 120 tokens per text instead of about 180. Shorter texts carry a weaker watermark.

## Qwen vs. OPT-1.3B: what's similar
- With no attack, KGW, PF and SynthID are still caught 98–100% of the time.
- KGW and SIR are still the most robust and also the most costly. Perplexity rises:
  - KGW: +53% on Qwen, +37% on OPT
  - SIR: +56% on Qwen, +48% on OPT
- PF is still the weakest under attack. Its lower perplexity again comes with slightly more repetitive text.
- DIPPER paraphrasing still removes most watermarks. SynthID ends at 14% on both models.
- SIR is still about 2x slower to generate.

## Qwen vs. OPT-1.3B: what's different
- **Weaker detection overall.** Even without an attack, MorphMark catches only 89% and SIR 80% (OPT: 98–100%).
- **Word deletion hurts much more.** On OPT, four methods kept 91–97%. On Qwen:
  - SIR and KGW keep about 75%
  - SynthID 58% and MorphMark 51%
  - PF 18%
- **SIR loses its lead under DIPPER.** It drops from 51% on OPT to 16% on Qwen, and KGW becomes the most robust at 23%.
- **MorphMark and SynthID now lower perplexity** by about 14–15%. On OPT there was no clear change.
- **Likely cause:** the shorter texts, plus an instruction-tuned model that is more sure of its next word, which leaves less room to hide a signal. The attacks were not harsher on Qwen: attacked texts were about as similar to the originals as on OPT.

## Task benchmarks on Qwen (change vs. unwatermarked)
- **GSM8K math (64.8% correct without a watermark):** KGW −16% and SIR −19%, both clear drops. PF −6%, SynthID +1% and MorphMark +5% are within noise.
- **Summarization (ROUGE-L):** KGW −7% and SIR −10%, both clear drops. The others are within noise.
- **Translation (BLEU):** only SIR drops clearly (−9%).
- **Overall:** SynthID and MorphMark cost nothing measurable on any task. KGW and SIR hurt math and summarization, and SIR also hurts translation.

"Clear drop" means the 95% confidence interval of the paired change (same prompts, bootstrap) does not include zero. All numbers come from `summary.csv`, `paired_deltas.csv` and `detection_summary.csv` in this folder.

## How the GSM8K accuracy change is computed

This is the x-axis of the Qwen Figure 2 in `../plots_tables.ipynb` (`rel_change_pct` for `gsm8k` in `paired_deltas.csv`). Recomputing KGW from the per-question files in `quality/gsm8k/` gives exactly the reported −16.05% with CI [−22.71%, −8.41%].

### From questions to the accuracy change

**Step 1: Pick the questions** (`wmbench/data.py`, `load_task`)
- The GSM8K test set (1,319 problems) is downloaded once from OpenAI's repository, at a fixed version that is checked against a known hash (`bootstrap.py`, `ensure_gsm8k`).
- The first 500 problems are used, the same 500 for every method.
- The "gold" answer is the number after `####` in the official solution, with commas removed (`gsm8k_gold`).

**Step 2: Build the prompt**
- Qwen is a chat model, so the prompt goes through its chat template: the question, then "Please reason step by step, and put your final answer within \boxed{}."
- The prompt is identical for every method.

**Step 3: Generate one answer per question per method** (`wmbench/generation.py`)
- Each method writes up to 512 new tokens, using plain sampling at temperature 1.
- Each generation gets its own random seed, computed from (run seed, task, question id, method name). Since the method name is part of it, the unwatermarked and KGW answers to the same question use different random draws. This is part of why 24–32% of answers flip between the unwatermarked and watermarked runs, which makes the intervals wide.

**Step 4: Pull out the final number** (`extract_math_answer`)
- First choice: the number in the last `\boxed{...}`.
- Otherwise: the first number after `####`.
- Otherwise: the last number in the text.
- Commas are removed, so `1,234` becomes `1234`.

**Step 5: Score each answer** (`metrics.math_answer_correct`)
- Correct (1) if the extracted number equals the gold number when compared as decimals, otherwise wrong (0).
- No number found also counts as wrong.

**Step 6: Pair the answers** (`report.py`, `paired_deltas`)
- Keep only questions that ran without error for both the unwatermarked model and the method. Here that is all 500.
- This gives two lists of 0s and 1s in the same question order: `b` (unwatermarked) and `a` (watermarked).

**Step 7: Compute the change**
- Accuracy without watermark = mean of `b`. For Qwen that is 0.648.
- Accuracy with watermark = mean of `a`. For KGW that is 0.544.
- Relative change = (0.544 / 0.648 − 1) × 100 = **−16.05%**. This is `rel_change_pct`, the x-value in Figure 2.
- The file also has an absolute change, `delta_vs_none`: mean of (a − b) = −0.104, i.e. −10.4 percentage points.

### The 95% CI (paired bootstrap)

The code is `metrics.paired_bootstrap_rel_ci`.

1. **Resample questions, keeping pairs together.** Draw 500 question indices at random, with replacement, so some questions appear twice and some not at all. Each drawn question brings both of its answers (`a[i]` and `b[i]`). That is what "paired" means here.
2. **Recompute the change on the resample:** (mean of resampled `a` / mean of resampled `b` − 1) × 100.
3. **Repeat 1,000 times.** This gives 1,000 plausible values of the change.
4. **Take the 2.5th and 97.5th percentiles.** These are the CI bounds, **[−22.7%, −8.4%]** for KGW.

### Details worth knowing
- **Same resamples for every method.** The bootstrap random generator always starts from seed 0, and every method has 500 questions, so all five methods are resampled with the same sets of questions. The intervals are reproducible and comparable across methods.
- **Why pairing matters.** Questions differ a lot in difficulty, and resampling pairs together keeps that difference out of the interval. Only questions where the two answers differ widen it.
- **Rare skip.** A resample whose unwatermarked accuracy is exactly 0 would be skipped, to avoid dividing by zero. That never happens at 65% accuracy.
- **Where the absolute interval comes from.** `ci_low` / `ci_high` in `paired_deltas.csv` are built the same way, but on the difference of means instead of the ratio.
