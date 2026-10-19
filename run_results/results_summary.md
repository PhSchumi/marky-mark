# Results summary: `opt13b/dipper_run`

`dipper_run` benchmarks five text watermarking methods on `facebook/opt-1.3b`. It measures three things: how much each watermark hurts text quality, how reliably it can be detected, and whether it survives two attacks.

## Watermarking methods
- **KGW**: the classic "green list" method, which nudges the model toward a secret subset of words.
- **MorphMark**: a variant of KGW that adjusts how strongly it nudges.
- **PF (Permute-and-Flip)**: watermarks through a keyed sampling rule.
- **SynthID**: Google's tournament-style sampling.
- **SIR**: a meaning-based watermark designed to survive rewording.
- **Baselines**: unwatermarked model output (`none`) and human-written text.

## Datasets
- **C4** (Colossal Clean Crawled Corpus): a large collection of cleaned English web pages, built by Google from Common Crawl. The model gets the start of a web text and writes a continuation. The real rest of the text is the human-written comparison. This is the main dataset, used for quality, detection and attacks.
- **WMT16 (German to English)**: news sentences from the 2016 Workshop on Machine Translation, each with a professional English translation. The model translates German sentences, given 5 example translations in the prompt. BLEU measures how closely its output matches the reference translation.
- **CNN/DailyMail**: news articles paired with short bullet-point highlights written by the journalists. The model summarizes the article (cut to 400 words). ROUGE-L measures how much the summary overlaps with those highlights.

Translation and summarization show whether a watermark hurts tasks that have a "right answer", not just free writing.

## What was tested
- **Text quality** on three tasks:
  - open-ended text continuation (C4, 500 prompts, up to 200 new tokens)
  - German-to-English translation (WMT16, 200 sentences, BLEU score)
  - news summarization (CNN/DailyMail, 100 articles, ROUGE-L score)
  - perplexity (how natural the text is, judged by the larger `opt-2.7b`) and word diversity for all of them
- **Detection** on C4 only: whether watermarked text can be told apart from human text and from unwatermarked model text. All three are cut to the same length before scoring.
- **Robustness** on C4 only: the watermarked texts were attacked, then checked again for the watermark.
  - **Word deletion**: 30% of the words removed at random.
  - **DIPPER**: an 11-billion-parameter model rewrites the text sentence by sentence.

## Main findings
- **Clean detection is near-perfect for all five.** Even with only 1 in 100 human texts wrongly flagged, they catch 98–100% of watermarked texts.
- **Word deletion is mostly survivable.** Four of the five still catch 91–97%. PF drops to 47%.
- **DIPPER breaks them all to a large extent.** The share still caught falls to:
  - SIR: 51%
  - KGW: 38%
  - MorphMark: 32%
  - SynthID: 14%
  - PF: 10%
- **Quality trade-off:**
  - KGW and SIR make text noticeably less natural.
  - MorphMark and SynthID stay close to unwatermarked text.
  - PF has the lowest perplexity but also the lowest word diversity, so it tends to repeat itself.

SemStamp isn't part of this run. Its results are in `opt13b/dipper_semstamp_run`.
