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
