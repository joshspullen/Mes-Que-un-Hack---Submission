"""Unit tests for dead ball v2 (stream 2), ball-gap evidence (stream 4) and touches (stream 3b).

    python tests/test_streams_v2.py

Small made-up worlds, no match data. The behaviour on real football is measured on external games
by tools/external_bench.py, tools/fit_dead.py, tools/fit_gaps.py and tools/check_touches.py.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mqh.dead import DeadConfig, tracking_dead_mask  # noqa: E402
from mqh.gaps import gap_evidence  # noqa: E402
from mqh.touches import touch_events  # noqa: E402

PITCH = (105.0, 68.0)
FPS = 25


def world(n, seed=0):
    """22 players jogging around the centre circle, the ball moving among them."""
    rng = np.random.default_rng(seed)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    players = np.zeros((n, 22, 2))
    pos = rng.uniform([30, 10], [75, 58], size=(22, 2))
    heading = rng.uniform(0, 2 * np.pi, 22)
    for t in range(n):
        heading += rng.normal(0, 0.05, 22)
        pos += 4.0 / FPS * np.column_stack([np.cos(heading), np.sin(heading)])
        pos = np.clip(pos, [5, 5], [100, 63])
        players[t] = pos
    ball = players[:, 3].copy() + 0.5
    return ball, players, team


def test_dead_v2_walks_and_out_of_play():
    n = 60 * FPS
    ball, players, team = world(n)
    a, b = 25 * FPS, 40 * FPS          # a 15 s stoppage
    # players walk (0.4 m/s) during the stoppage: freeze most of their motion
    for t in range(a, b):
        players[t] = players[a] + (players[t] - players[a]) * 0.1
    ball[a - 5:a] = [104.5, 30.0]      # the ball reaches the goal line, then the tracking loses it
    ball[a:b] = np.nan
    dead = tracking_dead_mask(ball / PITCH, players / PITCH, PITCH, FPS, n, cfg=DeadConfig())
    inside = dead[a + 3 * FPS:b - 2 * FPS].mean()
    outside = np.concatenate([dead[3 * FPS:a - 3 * FPS], dead[b + 3 * FPS:-3 * FPS]]).mean()
    assert inside > 0.8, inside
    assert outside < 0.2, outside


def test_gap_evidence_is_antisymmetric_and_bounded():
    n = 30 * FPS
    ball, players, team = world(n, seed=1)
    players[:, team == 0, 1] = 34 + (players[:, team == 0, 1] - 34) * 1.6   # home spread wide
    players[:, team == 1, 1] = 34 + (players[:, team == 1, 1] - 34) * 0.5   # away compact
    e = gap_evidence(ball / PITCH, players / PITCH, team, PITCH, FPS)
    e_swapped = gap_evidence(ball / PITCH, players / PITCH, 1 - team, PITCH, FPS)
    assert np.isfinite(e).all() and np.abs(e).max() <= 1.0
    assert e.mean() > 0.05, e.mean()          # the wide team is the one with the ball
    assert np.allclose(e, -e_swapped, atol=1e-9)


def test_touch_at_a_deflection_only():
    n = 200
    team = np.array([0, 1], dtype=np.int8)
    players = np.full((n, 2, 2), np.nan)
    players[:, 0] = [52.0, 34.0]       # a home player at the deflection point
    players[:, 1] = [20.0, 10.0]       # an away player far away
    ball = np.zeros((n, 2))
    t0 = 100
    for t in range(n):                 # 12 m/s along +x, then sent back along +y
        ball[t] = [52.0 - 12.0 * (t0 - t) / FPS, 34.0] if t <= t0 else [52.0, 34.0 + 12.0 * (t - t0) / FPS]
    frames, teams = touch_events(ball / PITCH, players / PITCH, team, PITCH, FPS)
    assert len(frames) == 1, frames
    assert abs(int(frames[0]) - t0) <= 2 and int(teams[0]) == 0, (frames, teams)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK")


if __name__ == "__main__":
    main()
