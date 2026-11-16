"""Unit tests for mqh/refine.py (snapping team changes onto touches). Pure numpy, no data.

    python tests/test_refine.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mqh.refine import snap_switches  # noqa: E402


def lab(s):
    return np.array(list(s), dtype="<U1")


def test_change_moves_back_onto_the_new_teams_touch():
    labels = lab("H" * 100 + "A" * 100)             # decoded change at 100
    out = snap_switches(labels, [90], [1], window=25)
    assert (out[:90] == "H").all() and (out[90:] == "A").all()


def test_change_moves_forward_onto_a_later_touch():
    labels = lab("H" * 100 + "A" * 100)
    out = snap_switches(labels, [112], [1], window=25)
    assert (out[:112] == "H").all() and (out[112:] == "A").all()


def test_only_the_new_teams_touches_count():
    labels = lab("H" * 100 + "A" * 100)
    out = snap_switches(labels, [95, 105], [0, 0], window=25)   # touches by the old team only
    assert (out == labels).all()


def test_nearest_touch_wins_and_window_is_respected():
    labels = lab("H" * 100 + "A" * 100)
    out = snap_switches(labels, [80, 97, 140], [1, 1, 1], window=25)
    assert (out[97:] == "A").all() and (out[:97] == "H").all()
    assert (snap_switches(labels, [60], [1], window=25) == labels).all()   # outside the window


def test_dead_edges_and_dead_frames_never_move():
    labels = lab("H" * 50 + "D" * 30 + "A" * 50 + "H" * 50)   # H D A | A H: only A->H (at 130) is direct
    out = snap_switches(labels, [45, 70, 120], [1, 1, 0], window=25)
    assert (out[50:80] == "D").all(), "dead frames were relabelled"
    assert (out[:50] == "H").all() and (out[80:120] == "A").all(), "a change across a stoppage moved"
    assert (out[120:] == "H").all(), "the direct A->H change did not snap to the home touch"


def test_a_change_never_crosses_another_change():
    labels = lab("H" * 50 + "A" * 10 + "H" * 50)             # short A run between two changes
    out = snap_switches(labels, [0, 100], [1, 1], window=200)
    assert (out == "A").sum() >= 1 and out[0] == "H" and out[-1] == "H"
    changes = int(np.sum(out[1:] != out[:-1]))
    assert changes == 2, changes


def test_no_touches_or_zero_window_change_nothing():
    labels = lab("H" * 100 + "A" * 100)
    assert (snap_switches(labels, [], [], 25) == labels).all()
    assert (snap_switches(labels, [90], [1], 0) == labels).all()


def test_input_is_not_modified_and_deterministic():
    labels = lab("H" * 100 + "A" * 100)
    before = labels.copy()
    a = snap_switches(labels, [90], [1])
    b = snap_switches(labels, [90], [1])
    assert (labels == before).all() and (a == b).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok   " + t.__name__)
    print("OK ({} tests)".format(len(tests)))
