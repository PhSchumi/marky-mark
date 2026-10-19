"""Datasets and task prompts.

All datasets ship inside the pinned MarkLLM checkout (``third_party/MarkLLM/dataset``):

* ``c4``            - C4 RealNewsLike prefixes with their human continuation (completion; PPL,
                      diversity, detection). MarkLLM paper protocol.
* ``wmt16_de_en``   - WMT16 German->English validation pairs (translation; BLEU).
* ``cnn_dailymail`` - CNN/DailyMail test articles with highlights (summarisation; ROUGE).
* ``humaneval``     - HumanEval problems (code; pass@1). Executes model code, off by default.

``gsm8k`` (grade-school math word problems; final-answer accuracy) is not part of MarkLLM: its
test split is downloaded once from OpenAI's repository by :func:`wmbench.bootstrap.ensure_gsm8k`.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from wmbench.bootstrap import ensure_gsm8k, markllm_dir


@dataclass
class Sample:
    id: str
    task_prompt: str  # text before chat formatting
    reference: str | dict | None = None  # BLEU/ROUGE reference or HumanEval test bundle
    human_text: str | None = None  # human-written continuation (detection negatives)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str  # completion | translation | summarization | code | math
    file: str
    quality_metrics: tuple[str, ...]
    max_new_tokens: int
    description: str


TASKS: dict[str, TaskSpec] = {
    "c4": TaskSpec("c4", "completion", "dataset/c4/processed_c4.json",
                   ("ppl", "log_diversity"), 200,
                   "Open-ended continuation of C4 news prefixes; detection is measured here."),
    "wmt16_de_en": TaskSpec("wmt16_de_en", "translation", "dataset/wmt16_de_en/validation.jsonl",
                            ("bleu",), 96, "German->English translation scored with sacreBLEU."),
    "cnn_dailymail": TaskSpec("cnn_dailymail", "summarization", "dataset/cnn_dailymail/test-00000-of-00001.jsonl",
                              ("rouge",), 96, "News summarisation scored with ROUGE-1/2/L."),
    "humaneval": TaskSpec("humaneval", "code", "dataset/human_eval/test.jsonl",
                          ("pass@1",), 300, "Python function completion scored by executing the unit tests."),
    "gsm8k": TaskSpec("gsm8k", "math", "gsm8k/test.jsonl", ("accuracy",), 512,
                      "Grade-school math word problems; the final number must match the gold answer."),
}

# ----------------------------------------------------------------------------- prompt templates

TRANSLATION_INSTRUCTION = "Translate the following German sentence into English. Reply with the English translation only."
SUMMARY_INSTRUCTION = "Summarize the following news article in two or three sentences."
CODE_INSTRUCTION = "Complete the following Python function. Reply with the code of the function body only."
COMPLETION_INSTRUCTION = "Continue the following text. Reply with the continuation only."
# Qwen2.5's recommended math prompt; the final answer ends up in \boxed{...}.
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

_TRANSLATION_SHOT = "German: {de}\nEnglish: {en}"


@dataclass
class PromptTemplates:
    """Task prompt construction for base vs chat models.

    Base models get few-shot / cue-word prompts (``German: ... English:``, ``TL;DR:``);
    chat models get an instruction and the tokenizer's chat template is applied later
    by :class:`wmbench.models.PromptFormatter`.
    """

    chat: bool = False
    translation_shots: int = 3
    article_max_words: int = 450

    def completion(self, prefix: str) -> str:
        return f"{COMPLETION_INSTRUCTION}\n\n{prefix}" if self.chat else prefix

    def translation(self, source: str, shots: list[tuple[str, str]]) -> str:
        if self.chat:
            return f"{TRANSLATION_INSTRUCTION}\n\n{source}"
        demo = "\n\n".join(_TRANSLATION_SHOT.format(de=de, en=en) for de, en in shots[: self.translation_shots])
        return (demo + "\n\n" if demo else "") + f"German: {source}\nEnglish:"

    def summarization(self, article: str) -> str:
        article = " ".join(article.split()[: self.article_max_words])
        if self.chat:
            return f"{SUMMARY_INSTRUCTION}\n\n{article}"
        return f"{article}\n\nTL;DR:"

    def code(self, problem: str) -> str:
        return f"{CODE_INSTRUCTION}\n\n```python\n{problem}\n```" if self.chat else problem

    def math(self, question: str) -> str:
        if self.chat:
            return f"{question}\n\n{MATH_INSTRUCTION}"
        return f"Question: {question}\nAnswer: Let's think step by step."


# ----------------------------------------------------------------------------- post-processing

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (nested braces allowed), or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    content = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content)
        content.append(ch)
        i += 1
    return None


def _first_number(text: str) -> str:
    match = _NUMBER.search(text)
    return match.group().replace(",", "") if match else ""


def _last_number(text: str) -> str:
    matches = _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else ""


def extract_math_answer(text: str) -> str:
    """Final number of a math solution: last ``\\boxed{}``, else after ``####``, else the last number.

    Commas are removed (``1,234`` -> ``1234``); returns ``""`` when no number is found.
    """
    text = text.split("\nQuestion:")[0]  # base models go on with the next few-shot style question
    boxed = _last_boxed(text)
    if boxed is not None and _first_number(boxed):
        return _first_number(boxed)
    if "####" in text:
        after = _first_number(text.split("####")[-1])
        if after:
            return after
    return _last_number(text)


def gsm8k_gold(answer: str) -> str:
    """Gold number of a GSM8K solution (the text after ``####``), commas removed."""
    return answer.split("####")[-1].strip().replace(",", "")


def postprocess(kind: str, continuation: str) -> str:
    """Trim a raw continuation to the task answer (first line for MT, first paragraph for summaries...).

    Code completions keep their leading indentation (a base model continues the function body
    directly after the docstring); only leading blank lines and fences are removed.
    """
    if kind == "translation":
        text = continuation.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = lines[0] if lines else ""
        # base-model few-shot prompts continue with another "German:" block
        for cue in ("German:", "Deutsch:"):
            if cue in text:
                text = text.split(cue)[0].strip()
        return text
    if kind == "summarization":
        paras = [p.strip() for p in continuation.strip().split("\n\n") if p.strip()]
        return paras[0] if paras else ""
    if kind == "code":
        text = continuation
        if "```" in text:  # chat models wrap code in fences
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            text = text[len("python"):] if text.startswith("python") else text
        from wmbench.metrics import clean_code_completion

        return clean_code_completion(text).rstrip()
    if kind == "math":
        return extract_math_answer(continuation)
    return continuation


# ----------------------------------------------------------------------------- loading


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_task(name: str, n_samples: int, templates: PromptTemplates | None = None, *,
              markllm_root: Path | None = None, shuffle_seed: int | None = None) -> list[Sample]:
    """Load ``n_samples`` prompts for a task.

    By default the first ``n_samples`` rows are used (MarkLLM convention). With
    ``shuffle_seed`` a seeded random subset is drawn instead. Few-shot demonstrations for
    translation are the *last* ``translation_shots`` rows of the (optionally shuffled) file,
    reserved before the evaluated slice is taken, so prompts do not depend on ``n_samples``.
    """
    spec = TASKS[name]
    templates = templates or PromptTemplates()
    if spec.kind == "math":
        path = ensure_gsm8k()
    else:
        path = (markllm_root or markllm_dir()) / spec.file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run wmbench.bootstrap.ensure_markllm() first")

    rows = _read_jsonl(path)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    shots: list[tuple[str, str]] = []
    if spec.kind == "translation" and templates.translation_shots > 0:
        k = templates.translation_shots
        shots = [(r["de"], r["en"]) for r in rows[-k:]]
        rows = rows[:-k]
    if n_samples > len(rows):
        raise ValueError(f"{name}: requested {n_samples} samples but only {len(rows)} are available")
    chosen = rows[:n_samples]
    samples: list[Sample] = []

    if spec.kind == "completion":
        for i, r in enumerate(chosen):
            samples.append(Sample(f"c4-{i}", templates.completion(r["prompt"]), None, r["natural_text"],
                                  {"prefix": r["prompt"]}))
    elif spec.kind == "translation":
        for i, r in enumerate(chosen):
            samples.append(Sample(f"wmt16-{i}", templates.translation(r["de"], shots), r["en"], r["en"],
                                  {"source": r["de"]}))
    elif spec.kind == "summarization":
        for i, r in enumerate(chosen):
            truncated = len(r["article"].split()) > templates.article_max_words
            samples.append(Sample(f"cnndm-{r.get('id', i)}", templates.summarization(r["article"]),
                                  r["highlights"], r["highlights"], {"prompt_truncated": truncated}))
    elif spec.kind == "code":
        for r in chosen:
            prompt = r["prompt"]
            samples.append(Sample(r["task_id"].replace("/", "-"), templates.code(prompt),
                                  {"task": prompt, "test": r["test"], "entry_point": r["entry_point"]},
                                  r.get("canonical_solution")))
    elif spec.kind == "math":
        for i, r in enumerate(chosen):
            samples.append(Sample(f"gsm8k-{i}", templates.math(r["question"]), gsm8k_gold(r["answer"]),
                                  None, {"solution": r["answer"]}))
    return samples
