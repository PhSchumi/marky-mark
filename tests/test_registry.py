import json
import math

from wmbench.attacks import word_deletion
from wmbench.config import DEFAULT_VARIANTS
from wmbench.watermarks import CONFIG_DIR, VARIANTS, WATERMARK_VARIANTS, _one_sided_normal_tail


def test_registry_is_consistent():
    assert set(DEFAULT_VARIANTS) <= set(VARIANTS)
    assert {"MorphMark", "PF", "SynthID", "SIR"} <= set(WATERMARK_VARIANTS)  # the four requested schemes
    for name, spec in VARIANTS.items():
        assert spec.name == name
        if spec.algorithm is None:
            assert spec.config_file is None
            continue
        path = CONFIG_DIR / spec.config_file
        assert path.exists(), path
        cfg = json.loads(path.read_text())
        assert cfg["algorithm_name"] == spec.algorithm
        assert spec.score_name and spec.citation
        assert spec.fixed_threshold is not None


def test_nominal_fpr_of_z_thresholds():
    assert VARIANTS["KGW"].nominal_fpr == _one_sided_normal_tail(4.0)
    assert math.isclose(_one_sided_normal_tail(2.0), 0.02275, rel_tol=1e-3)
    assert math.isclose(_one_sided_normal_tail(4.0), 3.167e-5, rel_tol=1e-2)
    assert VARIANTS["PF"].nominal_fpr == 0.01


def test_semstamp_restricted_to_c4():
    assert VARIANTS["SemStamp"].datasets == ("c4",)
    assert VARIANTS["KGW"].datasets is None


def test_word_deletion_seeded():
    text = " ".join(f"w{i}" for i in range(200))
    a, b, c = word_deletion(text, 0.3, 1), word_deletion(text, 0.3, 1), word_deletion(text, 0.3, 2)
    assert a == b and a != c
    assert 100 < len(a.split()) < 180
    assert word_deletion("", 0.3, 0) == ""
