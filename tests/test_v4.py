"""Engine v4: joint decoder, committed models, external loaders.

    python -m pytest tests/test_v4.py -q
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mqh.decoder3 import Decoder3Config, decode3, viterbi3  # noqa: E402


def _probs(seq, fps=25):
    """'H'/'A'/'D' seconds -> confident (p_home, p_dead) per frame."""
    p_home, p_dead = [], []
    for lab in seq:
        p_home += [0.9 if lab == "H" else 0.1 if lab == "A" else 0.5] * fps
        p_dead += [0.95 if lab == "D" else 0.05] * fps
    return np.array(p_home), np.array(p_dead)


def test_clean_evidence_is_followed():
    seq = "HHHHHAAAAADDDDHHHH"
    p_home, p_dead = _probs(seq)
    n = len(p_home)
    labels = decode3(p_home, p_dead, [(1, 0, n - 1)], n)
    expected = np.repeat(list(seq), 25)
    assert (labels == expected).mean() > 0.97


def test_short_flicker_is_smoothed():
    p_home, p_dead = _probs("HHHHHHHHHH")
    p_home[100:105] = 0.1                    # 0.2 s of contrary evidence
    assert set(viterbi3(p_home, p_dead).tolist()) == {0}


def test_outside_blocks_is_x_and_pre_kickoff_is_dead():
    p_home, p_dead = _probs("HHHHAAAA")
    n = len(p_home)
    labels = decode3(p_home, p_dead, [(1, 25, n - 1)], n, pre_kickoff=(25, 74))
    assert (labels[:25] == "X").all()
    assert (labels[25:75] == "D").all()


def test_decoder_is_deterministic():
    rng = np.random.default_rng(0)
    p_home, p_dead = rng.random(5000), rng.random(5000)
    a = decode3(p_home, p_dead, [(1, 0, 4999)], 5000, Decoder3Config())
    b = decode3(p_home.copy(), p_dead.copy(), [(1, 0, 4999)], 5000, Decoder3Config())
    assert (a == b).all()


def test_committed_models_match_manifest():
    models = os.path.join(ROOT, "models")
    with open(os.path.join(models, "MANIFEST.json")) as handle:
        manifest = json.load(handle)
    for name, digest in manifest["files"].items():
        with open(os.path.join(models, name), "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == digest, name


def test_models_load_and_predict():
    import sklearn
    from mqh import dead_model
    from mqh.v4 import load_models
    with open(os.path.join(ROOT, "models", "MANIFEST.json")) as handle:
        assert json.load(handle)["sklearn"] == sklearn.__version__
    tm, dm = load_models()
    n_features = dm.n_features_in_ if hasattr(dm, "n_features_in_") else 45
    p = dead_model.probability(dm, np.zeros((10, n_features), dtype=np.float32))
    assert p.shape == (10,) and np.all((p >= 0) & (p <= 1))
