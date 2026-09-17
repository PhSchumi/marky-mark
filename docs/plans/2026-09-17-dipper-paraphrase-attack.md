# DIPPER paraphrase attack: implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a paraphrase attack (DIPPER, Krishna et al. 2023) as a second robustness attack next to word
deletion, measured on the length-controlled watermarked C4 texts with attack strength reported.

**Architecture:** A new stage `attack` (between `generate` and `detect`) loads one attacker at a time, attacks
the common-budget watermarked text of every (variant, sample) and caches the result with strength metrics in
`attacks/<dataset>/<attack>/<variant>.jsonl`. Stage `detect` scores those texts as `attacked:<name>` rows; the
report adds per-attack columns. DIPPER is re-implemented as a small adapter (same procedure and settings as
MarkLLM's `DipperParaphraser`) to avoid MarkLLM's evaluation import chain (`translate`, OpenAI).

**Status:** implemented 2026-09-17 (all six tasks; 71 fast + 3 slow tests).

**Tech stack:** transformers T5ForConditionalGeneration (`kalpeshk2011/dipper-paraphraser-xxl`, tokenizer
`google/t5-v1_1-xxl`, bf16 on CUDA), nltk punkt, sacrebleu, sentence-transformers (optional similarity).

---

## Decisions

* Attack input = the watermarked text cut to the sample's common budget `T` (the text scored clean), prefix =
  the raw C4 prompt. Output is capped at `T` tokens so a longer paraphrase cannot add evidence; the uncapped
  length is stored. Negatives stay clean (threshold calibrated on unattacked human text).
* Strength per row: `len_ratio` (uncapped tokens / T), `bleu_vs_source` (sacreBLEU, model-free) and
  `sim_embed` (cosine similarity of sentence embeddings; `None` when the model is unavailable).
* Seed per attack row: `stable_seed(seed, "attack", attack, dataset, sample_id)`, same across variants.
* `attack_config_hash` (attacker id, revision, settings) per row; resume needs same hash and same `T`.
* Attack settings are excluded from the run `config_hash` (row-level hash handles regeneration);
  `word_deletion_ratio` stays in the run hash so existing run directories keep their hash.
* DIPPER defaults follow MarkLLM: lex 60, order 0, sent_interval 1, `do_sample, top_p 0.75, top_k 0,
  max_new_tokens 512`. Smoke preset runs `attacks=("word_deletion",)` only.

## Tasks

### Task 1: attack registry and attackers (`wmbench/attacks.py`, `tests/test_attacks.py`)
`AttackSpec`/`ATTACKS`, `Attacker` protocol (`name`, `config()`, `attack(text, prompt, seed) -> str`, `close()`),
`WordDeletionAttacker`, `DipperAttacker` (+ `from_pretrained(cfg, device)`), `build_attacker(name, cfg, device)`,
`attack_config_digest(attacker)`. Tests with a fake seq2seq model/tokenizer: prompt format
`lexical = 40, order = 100 <prefix> <sent> s </sent>`, one generate call per sentence window, growing prefix,
empty text -> "", diversity validation.

### Task 2: config + stage `attack` (`wmbench/config.py`, `wmbench/evaluate.py`, `tests/test_attack_stage.py`)
Config fields (`attacks`, `dipper_*`, `attack_similarity_model`, hash exclusions, smoke preset). `attack_path()`,
`attack_records(attacker, tokenizer, dataset, records, none_by_id, cfg, env, store, similarity=None, ...)`,
`Runner.stage_attack()`, STAGES with `attack`. Tests: rows carry `detect_tokens == T`, capped `n_tokens <= T`,
`len_ratio`, `bleu_vs_source`, resume/re-attack on `T` change, skip without `none`.

### Task 3: detect integration (`wmbench/evaluate.py`, `tests/test_detect.py`)
`detect_records(..., attacks=...)`: kinds `attacked:<name>` from attack rows with matching `T`; row carries
`attack_config_hash`; resume also keyed on it; inline word deletion removed.

### Task 4: report (`wmbench/report.py`, `tests/test_report.py`)
Load `attacks/` frame; per attack: `attacked_n_*`, `attacked_tpr_at_clean_thr_*`, `attacked_auroc_human_*`,
`attacked_fixed_thr_tpr_*`, `attacked_tokens_mean_*`, `attack_len_ratio_*`, `attack_bleu_vs_source_*`,
`attack_sim_embed_*`; main table `attacked_tpr_*`; histogram colours for attacked kinds.

### Task 5: prefetch, CLI, SLURM, docs, notebook
`prepare_assets.py --dipper`, `--attacks` / `--dipper-model` flags + env, `scripts/slurm/attack.sbatch`,
README (stages, metrics, robustness section), notebook builder section + rebuild.

### Task 6: integration (`tests/test_integration_slow.py`)
Tiny random T5 + OPT tokenizer through `DipperAttacker` -> `attack_records` -> `detect_records` with KGW;
`test_smoke_pipeline` with `run_robustness=True, attacks=("word_deletion",)`.
