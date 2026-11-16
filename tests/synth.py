"""A made-up match in the real file formats, with known possessions. No real data involved.

    python tests/synth.py <out_dir>

writes  <out_dir>/synth_atd.txt, synth_atd.xml, synth_gt.csv  (full video length, real half
boundaries). Possessions alternate between teams; the ball sits at the carrier's feet, goes
missing in long gaps (~40% of frames), and dead-ball stretches alternate between "ball out of
bounds" and "ball still on the pitch".
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games import GAMES, GAME, half_video_blocks  # noqa: E402

LENGTH, WIDTH = GAMES[GAME]["pitch"]
N = GAMES[GAME]["n_video_frames"]
N_SLOTS = 46
KICKOFF_2 = 113742


def make(seed=7, turnover_frac=0.0, dead_flip_prob=1.0):
    rng = np.random.RandomState(seed)
    truth = np.full(N, "X", dtype="<U1")
    ball = np.full((N, 2), np.nan)
    players = np.full((N, N_SLOTS, 2), np.nan)

    # 11 home slots (0-10) and 11 away slots (23-33); the rest stay empty, like spare tracks
    home_slots, away_slots = np.arange(0, 11), np.arange(23, 34)
    base_h = np.column_stack([rng.uniform(0.1, 0.9, 11), rng.uniform(0.1, 0.9, 11)])
    base_a = np.column_stack([rng.uniform(0.1, 0.9, 11), rng.uniform(0.1, 0.9, 11)])

    for half, first, last in half_video_blocks(GAME):
        t = first
        if half == 2:
            truth[first:KICKOFF_2] = "D"
            ball[first:KICKOFF_2] = (0.5, 0.5)
            t = KICKOFF_2
        team = 0
        while t <= last:
            dur = int(rng.uniform(100, 700))
            end = min(last + 1, t + dur)
            truth[t:end] = "HA"[team]
            n = end - t
            phase = np.arange(n) / 90.0
            ph = rng.uniform(0, 6, 22)
            for k, s in enumerate(home_slots):
                players[t:end, s, 0] = base_h[k, 0] + 0.03 * np.sin(phase + ph[k])
                players[t:end, s, 1] = base_h[k, 1] + 0.03 * np.cos(phase + ph[k])
            for k, s in enumerate(away_slots):
                players[t:end, s, 0] = base_a[k, 0] + 0.03 * np.sin(phase + ph[11 + k])
                players[t:end, s, 1] = base_a[k, 1] + 0.03 * np.cos(phase + ph[11 + k])
            # the carrier changes every 2-4 s; the ball stays within ~0.6 m of the carrier
            seg = 0
            while seg < n:
                step = int(rng.uniform(50, 100))
                slot = rng.choice(home_slots if team == 0 else away_slots)
                a, b = t + seg, min(end, t + seg + step)
                off = rng.normal(0, 0.4, (b - a, 2)) / np.array([LENGTH, WIDTH])
                ball[a:b] = players[a:b, slot] + off
                seg += step
            t = end
            # dead stretch between possessions, except right at the end of the half
            if t <= last and (turnover_frac <= 0 or rng.rand() >= turnover_frac):
                dd = int(rng.uniform(75, 200))
                de = min(last + 1, t + dd)
                truth[t:de] = "D"
                if rng.rand() < 0.5:
                    ball[t:de] = (rng.choice([-0.01, 1.01]), rng.uniform(0.2, 0.8))  # out
                else:
                    ball[t:de] = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))      # still
                    ball[t:de] += rng.normal(0, 0.00002, (de - t, 2))
                players[t:de] = players[t - 1][None]
                t = de
                if dead_flip_prob >= 1 or rng.rand() < dead_flip_prob:
                    team = 1 - team      # else the same team keeps the ball after the stoppage
            else:
                team = 1 - team

    # tracking dropouts: long ball gaps (never on dead frames, so dead rules stay observable)
    live = np.isin(truth, ["H", "A"])
    pos = np.where(live)[0]
    covered = 0
    target = int(0.4 * len(pos))
    while covered < target:
        a = int(rng.choice(pos))
        length = int(rng.uniform(100, 700))
        ball[a:a + length] = np.nan
        covered += length
    return truth, ball, players


def write(out_dir, turnover_frac=0.0, seed=7, dead_flip_prob=1.0):
    os.makedirs(out_dir, exist_ok=True)
    truth, ball, players = make(seed=seed, turnover_frac=turnover_frac, dead_flip_prob=dead_flip_prob)

    with open(os.path.join(out_dir, "synth_atd.txt"), "w") as f:
        for i in range(N):
            if np.isnan(players[i, :, 0]).all() and np.isnan(ball[i, 0]):
                pl = ";".join(["NaN,NaN,NaN"] * N_SLOTS)
                bl = "NaN,NaN,NaN"
            else:
                pl = ";".join(
                    "NaN,NaN,NaN" if np.isnan(p[0]) else "{:.5f},{:.5f},0.5".format(p[0], p[1])
                    for p in players[i])
                bl = ("NaN,NaN,NaN" if np.isnan(ball[i, 0])
                      else "{:.5f},{:.5f},NaN".format(ball[i, 0], ball[i, 1]))
            f.write("{}:{}:{}\n".format(i + 1, pl, bl))

    slot_team = ["A"] * 23 + ["B"] * 23
    xml = ["<root>", '<Team id="A"><Name>Home</Name></Team>', '<Team id="B"><Name>Away</Name></Team>']
    xml += ['<Player teamId="{}"/>'.format(t) for t in slot_team]
    xml.append("</root>")
    with open(os.path.join(out_dir, "synth_atd.xml"), "w") as f:
        f.write("\n".join(xml))

    with open(os.path.join(out_dir, "synth_gt.csv"), "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(truth))
    return truth


if __name__ == "__main__":
    write(sys.argv[1] if len(sys.argv) > 1 else "synth_out")
