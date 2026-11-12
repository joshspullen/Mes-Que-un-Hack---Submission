"""Unit checks for mqh.decode.team_evidence on tiny hand-built scenes (no real data).

    python tests/test_evidence.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mqh.decode import EvidenceConfig, smoothed_ball_speed, team_evidence  # noqa: E402

PITCH = (105.0, 68.0)
FPS = 25
TEAM = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)  # slots 0-2 home, 3-5 away
FAR = (0.95, 0.95)


def norm(x_m, y_m):
    return (x_m / PITCH[0], y_m / PITCH[1])


def scene(n, ball_m, players_m):
    """Static scene repeated over n frames. ball_m: (n,2) or (2,) metres; players_m: {slot: (x,y)}."""
    ball = np.broadcast_to(np.asarray(ball_m, float), (n, 2)) / np.array(PITCH)
    players = np.full((n, len(TEAM), 2), np.nan)
    for slot, xy in players_m.items():
        players[:, slot] = np.asarray(xy, float) / np.array(PITCH)
    return np.array(ball), players


def ev(ball, players, cfg=EvidenceConfig()):
    return team_evidence(ball, players, TEAM, PITCH, cfg, fps=FPS)


def test_sign_and_gap():
    ball, players = scene(10, (50, 30), {0: (50.5, 30), 3: (55, 30)})
    ball[4:6] = np.nan
    e = ev(ball, players)
    assert (e[[0, 1, 2, 3, 6, 7, 8, 9]] > 0.2).all(), e
    assert (e[4:6] == 0).all(), "no ball must give exactly 0"
    players[:, [0, 3]] = players[:, [3, 0]]
    assert (ev(ball, players)[ball[:, 0] == ball[:, 0]] < -0.2).all()


def test_flying_ball_is_downweighted():
    n = 40
    players = {0: (50.5, 30), 3: (54, 30)}
    still, pl = scene(n, (50, 30), players)
    x = 50 + np.arange(n) * 20.0 / FPS  # 20 m/s pass, starting next to the same players
    flying = np.column_stack([x, np.full(n, 30.0)]) / np.array(PITCH)
    v = smoothed_ball_speed(flying, PITCH, FPS)
    assert np.allclose(v[3:-3], 20.0, atol=0.1), v
    e_still, e_fly = ev(still, pl)[:3], ev(flying, pl)[:3]
    assert (e_fly < 0.3 * e_still).all(), (e_still, e_fly)
    assert (e_fly > 0).all(), "a flying ball keeps a small vote (speed_floor)"


def test_jitter_is_not_a_pass():
    rng = np.random.RandomState(0)
    n = 200
    jitter = rng.normal(0, 0.15, (n, 2)) + (50, 30)  # 15 cm tracking noise on a still ball
    v = smoothed_ball_speed(jitter / np.array(PITCH), PITCH, FPS)
    assert np.nanmedian(v) < 5.0, np.nanmedian(v)


def test_nearby_count_matters():
    # same nearest-player margin, but away has two more players around the ball
    base = {0: (51, 30), 3: (52, 30)}
    ball, p1 = scene(5, (50, 30), base)
    _, p2 = scene(5, (50, 30), {**base, 4: (50, 32.5), 5: (50, 27.5)})
    assert (ev(ball, p2) < ev(ball, p1)).all()


def test_contested_ball_is_damped():
    # a duel: home 0.5 m and away 0.8 m from the ball, with and without the contest term
    ball, pl = scene(5, (50, 30), {0: (50.5, 30), 3: (49.2, 30)})
    on, off = ev(ball, pl), ev(ball, pl, EvidenceConfig(contest_floor=1.0))
    assert (np.abs(on) < 0.7 * np.abs(off)).all(), (on, off)
    # an uncontested ball (away 8 m) is barely touched by the contest term
    ball, pl = scene(5, (50, 30), {0: (50.5, 30), 3: (58, 30)})
    on, off = ev(ball, pl), ev(ball, pl, EvidenceConfig(contest_floor=1.0))
    assert np.allclose(on, off, rtol=0.01)


def test_range():
    rng = np.random.RandomState(1)
    n = 500
    ball = rng.uniform(0, 1, (n, 2))
    ball[rng.rand(n) < 0.3] = np.nan
    players = rng.uniform(0, 1, (n, len(TEAM), 2))
    players[rng.rand(n, len(TEAM)) < 0.2] = np.nan
    e = ev(ball, players)
    assert np.isfinite(e).all() and (np.abs(e) <= 1).all()


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("OK")


if __name__ == "__main__":
    main()
