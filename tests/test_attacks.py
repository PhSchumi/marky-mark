"""Attack registry and attackers (fake models; no downloads)."""

from __future__ import annotations

import pytest
import torch

from wmbench.attacks import (
    ATTACKS,
    DipperAttacker,
    WordDeletionAttacker,
    attack_config_digest,
    word_deletion,
)


class FakeSeq2SeqTokenizer:
    """Records every prompt; ids are just character codes."""

    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, texts, return_tensors="pt"):
        self.prompts.extend(texts)
        ids = [[ord(c) for c in t] for t in texts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), len(ids[0]), dtype=torch.long)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(int(i)) for i in row) for row in ids]


class FakeSeq2SeqModel:
    """Echoes the text between <sent> and </sent> in upper case; counts calls; has a device."""

    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0
        self.kwargs = None

    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        text = "".join(chr(int(i)) for i in input_ids[0])
        inner = text.split("<sent> ", 1)[1].rsplit(" </sent>", 1)[0]
        return torch.tensor([[ord(c) for c in inner.upper()]])


def _dipper(**kw):
    tok, model = FakeSeq2SeqTokenizer(), FakeSeq2SeqModel()
    base = dict(model_name="fake/dipper", tokenizer_name="fake/t5", revision="abc", lex_diversity=60,
                order_diversity=0, sent_interval=1)
    return DipperAttacker(model, tok, **{**base, **kw}), tok, model


def test_registry_lists_both_attacks():
    assert set(ATTACKS) >= {"word_deletion", "dipper"}
    assert ATTACKS["dipper"].needs_model and not ATTACKS["word_deletion"].needs_model
    for spec in ATTACKS.values():
        assert spec.description and spec.citation


def test_word_deletion_attacker_matches_the_function_and_is_seeded():
    a = WordDeletionAttacker(0.3)
    text = " ".join(f"w{i}" for i in range(50))
    assert a.attack(text, "prompt", seed=5) == word_deletion(text, 0.3, 5)
    assert a.attack(text, "prompt", seed=5) == a.attack(text, "prompt", seed=5)
    assert a.name == "word_deletion" and a.config() == {"attack": "word_deletion", "ratio": 0.3}


def test_dipper_prompt_format_and_sentence_windows():
    a, tok, model = _dipper(lex_diversity=60, order_diversity=0, sent_interval=1)
    out = a.attack("First sentence here. Second one follows! Third?", "The prompt\nwith newline", seed=1)
    assert model.calls == 3
    assert tok.prompts[0] == "lexical = 40, order = 100 The prompt with newline <sent> First sentence here. </sent>"
    # the paraphrased output is appended to the prefix of the next window (MarkLLM / DIPPER procedure)
    assert tok.prompts[1] == ("lexical = 40, order = 100 The prompt with newline FIRST SENTENCE HERE. "
                              "<sent> Second one follows! </sent>")
    assert out == "FIRST SENTENCE HERE. SECOND ONE FOLLOWS! THIRD?"
    assert model.kwargs["do_sample"] is True and model.kwargs["top_p"] == 0.75 and model.kwargs["top_k"] == 0


def test_dipper_sentence_interval_groups_sentences():
    a, tok, model = _dipper(sent_interval=2)
    a.attack("A one. B two. C three.", "", seed=1)
    assert model.calls == 2
    assert tok.prompts[0] == "lexical = 40, order = 100 <sent> A one. B two. </sent>"


def test_dipper_empty_text_makes_no_model_call():
    a, tok, model = _dipper()
    assert a.attack("   ", "prompt", seed=1) == "" and model.calls == 0


def test_dipper_rejects_invalid_diversity():
    with pytest.raises(ValueError, match="diversity"):
        _dipper(lex_diversity=50)


def test_attack_config_digest_changes_with_settings_and_model():
    a, _, _ = _dipper()
    b, _, _ = _dipper(lex_diversity=80)
    c, _, _ = _dipper(revision="def")
    assert a.config()["attack"] == "dipper" and a.config()["model"] == "fake/dipper"
    assert len({attack_config_digest(a), attack_config_digest(b), attack_config_digest(c)}) == 3
    assert attack_config_digest(a) == attack_config_digest(_dipper()[0])
