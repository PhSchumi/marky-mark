"""Shared fakes for the model-free tests (tokenizer, watermark variant)."""

from __future__ import annotations

from wmbench.watermarks import DetectionOutput


class FakeTokenizer:
    """Whitespace tokenizer with a stable word -> id table (decode(encode(x)) == x)."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.words)
                self.words.append(w)
            ids.append(self.vocab[w])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[i] for i in ids)


class FakeVariant:
    """Detector that reports the token count it saw and remembers every scored text."""

    def __init__(self, tok, name="KGW"):
        self.tok, self.name, self.seen = tok, name, []

    def effective_config(self):
        return {"algorithm": self.name, "delta": 2.0}

    def detect(self, text):
        n = len(self.tok(text)["input_ids"])
        self.seen.append(text)
        if n == 0:
            return DetectionOutput(float("nan"), None, 0, "empty text", {})
        return DetectionOutput(float(n), n > 3, n, None, {"text": text})


def words(n, tag):
    return " ".join(f"{tag}{i}" for i in range(n))
