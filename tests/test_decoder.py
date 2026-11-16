"""Unit tests for mqh/decoder.py. Pure numpy, runs in about a second, no data.

    python tests/test_decoder.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mqh.decoder import (DecoderConfig, clean_dead_mask, decode_half,  # noqa: E402
                         viterbi_two_state)


def decode(ev, dead=None, **kw):
    ev = np.asarray(ev, dtype=float)
    dead = np.zeros(len(ev), dtype=bool) if dead is None else np.asarray(dead, dtype=bool)
    return decode_half(ev, dead, 0, len(ev) - 1, DecoderConfig(**kw))


def switches(labels):
    live = labels != "D"
    prev = None
    out = []
    for i, lab in enumerate(labels):
        if lab == "D":
            continue
        if prev is not None and lab != prev:
            out.append(i)
        prev = lab
    return out


def test_penalty_suppresses_flicker():
    rng = np.random.RandomState(0)
    ev = 0.2 + rng.normal(0, 0.6, 600)          # mildly home, very noisy
    flicker = viterbi_two_state(ev, 0.0)
    smooth = viterbi_two_state(ev, 30.0)
    n_flicker = int(np.sum(flicker[1:] != flicker[:-1]))
    n_smooth = int(np.sum(smooth[1:] != smooth[:-1]))
    assert n_flicker > 100 and n_smooth <= 6, (n_flicker, n_smooth)


def test_minimum_run_length_follows_penalty():
    # a burst of away evidence shorter than the penalty can never flip the state, longer can
    ev = np.concatenate([np.ones(100), -np.ones(20), np.ones(100)])
    assert switches(decode(ev, switch_penalty=30.0, place_switch_in_gaps=False)) == []
    ev = np.concatenate([np.ones(100), -np.ones(60), np.ones(100)])
    assert len(switches(decode(ev, switch_penalty=30.0, place_switch_in_gaps=False))) == 2


def test_handover_inside_silent_gap_goes_to_midpoint():
    # home evidence 0-99, silence 100-299, away evidence 300-399: true switch somewhere in 100-300
    ev = np.concatenate([np.ones(100), np.zeros(200), -np.ones(100)])
    mid = switches(decode(ev))
    assert len(mid) == 1 and abs(mid[0] - 200) <= 1, mid
    raw = switches(decode(ev, place_switch_in_gaps=False))
    assert len(raw) == 1 and abs(raw[0] - 200) > 50, "tie-break should be arbitrary: {}".format(raw)


def test_handover_across_a_stoppage_goes_to_the_restart():
    # home 0-99, silent+dead 100-149, silent live 150-189, away evidence 190-289
    ev = np.concatenate([np.ones(100), np.zeros(90), -np.ones(100)])
    dead = np.zeros(len(ev), dtype=bool)
    dead[100:150] = True
    labels = decode(ev, dead)
    assert (labels[100:150] == "D").all()
    assert switches(labels) == [150], switches(labels)
    assert (labels[:100] == "H").all() and (labels[150:] == "A").all()


def test_cheaper_switch_after_a_stoppage():
    # Each frame of contrary evidence e swings the score by 2*|e|. 25 frames of -0.4 are worth 20:
    # not enough to beat a penalty of 30, enough to beat the 15 it becomes after a stoppage.
    ev = np.concatenate([np.ones(100), np.zeros(30), -0.4 * np.ones(25)])
    assert switches(decode(ev, switch_penalty=30.0)) == []        # no stoppage: no switch
    dead = np.zeros(len(ev), dtype=bool)
    dead[100:130] = True                                          # 30 dead frames (>= 25)
    labels = decode(ev, dead, switch_penalty=30.0)
    assert switches(labels) == [130], switches(labels)            # restart discount applies
    dead = np.zeros(len(ev), dtype=bool)
    dead[100:110] = True                                          # a 10-frame blip is no stoppage
    assert switches(decode(ev, dead, switch_penalty=30.0)) == []


def test_clean_dead_mask():
    m = np.zeros(400, dtype=bool)
    m[50:53] = True            # 3-frame dead blip inside play  -> removed
    m[100:200] = True          # real stoppage
    m[210:260] = True          # 10 live frames, then another real stoppage
    m[300:310] = True          # 10-frame dead run (< 25)       -> removed
    out = clean_dead_mask(m, 25, 25)
    assert not out[50:53].any(), "dead blip survived"
    assert not out[300:310].any(), "short dead run survived"
    assert out[100:260].all(), "short live gap between stoppages was not filled"
    assert out.sum() == 160
    assert clean_dead_mask(np.zeros(10, dtype=bool), 25, 25).sum() == 0
    assert clean_dead_mask(np.ones(10, dtype=bool), 25, 25).sum() == 0  # a 10-frame mask is noise


def test_dead_blip_does_not_split_a_possession():
    ev = np.ones(300)
    dead = np.zeros(300, dtype=bool)
    dead[100] = True                                   # one stray dead frame
    cleaned = decode(ev, dead)
    assert (cleaned == "H").all(), "possession was split by a one-frame dead blip"
    kept = decode(ev, dead, clean_dead_mask=False)
    assert kept[100] == "D", "with cleaning off the raw mask must pass through"


def test_integer_decoding_ignores_float_noise():
    rng = np.random.RandomState(5)
    ev = np.clip(rng.normal(0, 0.5, 5000), -1, 1)
    dead = rng.rand(5000) < 0.03
    base = decode(ev, dead)
    for eps in (1e-12, -1e-12, 3e-13):
        assert (decode(ev + eps, dead) == base).all(), "last-digit float noise changed the labels"


def test_all_dead_and_empty_halves():
    labels = decode(np.zeros(50), np.ones(50, dtype=bool))
    assert (labels == "D").all()
    assert viterbi_two_state(np.zeros(0), 30.0).shape == (0,)


def test_two_possessions_around_a_stoppage_stay_two():
    ev = np.ones(300)
    dead = np.zeros(300, dtype=bool)
    dead[100:160] = True
    labels = decode(ev, dead)
    runs = [labels[0], labels[100], labels[160]]
    assert runs == ["H", "D", "H"]


def test_deterministic():
    rng = np.random.RandomState(3)
    ev = rng.normal(0, 0.5, 2000)
    dead = rng.rand(2000) < 0.05
    a, b = decode(ev, dead), decode(ev, dead)
    assert (a == b).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok   " + t.__name__)
    print("OK ({} tests)".format(len(tests)))
