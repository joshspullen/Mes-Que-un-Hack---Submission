"""Stream 5b: joint H / A / D decoding with a soft dead-ball probability.

The two-step decoder (mqh/decoder.py) first thresholds the dead-ball probability into a mask, then
decodes the team inside the live stretches. A wrong threshold decision is final: a dead frame the
mask missed splits nothing, a false dead frame splits a possession in two. Here both questions are
decided together, by one Viterbi pass over three states per frame:

    score(H) = log(1 - p_dead) + log p_home        score(D) = log p_dead
    score(A) = log(1 - p_dead) + log(1 - p_home)

with three transition costs: changing team inside play (`team_cost`), play -> stoppage
(`stop_cost`) and stoppage -> play (`restart_cost`). After a stoppage either team may restart at
no extra cost, so the decoder can hand the ball over at a restart without paying `team_cost`, which
is how most possession changes at a stoppage happen (a throw-in to the other side). Costs are in
frames' worth of log-odds and are chosen on external matches (tools/bench_all.py).

Durations: a run shorter than `min_dead_frames` / `min_live_frames` after decoding is merged into
its neighbours (a real stoppage lasts at least about a second, and so does a spell of play between
two stoppages).

Integers throughout (log-probabilities rounded to 1/SCALE), so the same input gives the same labels
on every machine.
"""

from dataclasses import dataclass

import numpy as np

SCALE = 1000
H, A, D = 0, 1, 2


@dataclass(frozen=True)
class Decoder3Config:
    team_cost: float = 10.0
    stop_cost: float = 4.0
    restart_cost: float = 4.0
    min_dead_frames: int = 25
    min_live_frames: int = 25
    p_floor: float = 0.01


def _ints(x):
    return np.rint(np.asarray(x, dtype=float) * SCALE).astype(np.int64)


def viterbi3(p_home, p_dead, cfg=Decoder3Config()):
    """Most likely H/A/D path (int8 array, 0 = H, 1 = A, 2 = D) for one half."""
    n = len(p_home)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    f = cfg.p_floor
    ph = np.clip(p_home, f, 1 - f)
    pd = np.clip(p_dead, f, 1 - f)
    eh = _ints(np.log1p(-pd) + np.log(ph)).tolist()
    ea = _ints(np.log1p(-pd) + np.log1p(-ph)).tolist()
    ed = _ints(np.log(pd)).tolist()
    ct = int(round(cfg.team_cost * SCALE))
    cs = int(round(cfg.stop_cost * SCALE))
    cr = int(round(cfg.restart_cost * SCALE))
    back = np.zeros((n, 3), dtype=np.int8)
    sh, sa, sd = eh[0], ea[0], ed[0]
    bh = back[:, 0]
    ba = back[:, 1]
    bd = back[:, 2]
    for t in range(1, n):
        # into H: from H, A (team change), D (restart)
        x, y, z = sh, sa - ct, sd - cr
        if x >= y and x >= z:
            nh, bh[t] = x, 0
        elif y >= z:
            nh, bh[t] = y, 1
        else:
            nh, bh[t] = z, 2
        x, y, z = sa, sh - ct, sd - cr
        if x >= y and x >= z:
            na, ba[t] = x, 1
        elif y >= z:
            na, ba[t] = y, 0
        else:
            na, ba[t] = z, 2
        x, y, z = sd, sh - cs, sa - cs
        if x >= y and x >= z:
            nd, bd[t] = x, 2
        elif y >= z:
            nd, bd[t] = y, 0
        else:
            nd, bd[t] = z, 1
        sh, sa, sd = nh + eh[t], na + ea[t], nd + ed[t]
    path = np.zeros(n, dtype=np.int8)
    s = int(np.argmax([sh, sa, sd]))   # ties -> H, A, D order: deterministic
    for t in range(n - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


def _runs(values):
    change = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(values)]))
    return starts, ends


def clean_durations(path, cfg=Decoder3Config()):
    """Merge dead runs shorter than min_dead_frames and live spells (between two stoppages)
    shorter than min_live_frames into their surroundings."""
    path = path.copy()
    dead = path == D
    s, e = _runs(dead.astype(np.int8))
    for a, b in zip(s, e):
        if dead[a] and b - a < cfg.min_dead_frames and a > 0 and b < len(path):
            fill = path[a - 1]
            path[a:b] = fill
    dead = path == D
    s, e = _runs(dead.astype(np.int8))
    for a, b in zip(s, e):
        if not dead[a] and b - a < cfg.min_live_frames and a > 0 and b < len(path):
            path[a:b] = D
    return path


def decode3(p_home, p_dead, blocks, n_frames, cfg=Decoder3Config(), pre_kickoff=None):
    """Labels 'H'/'A'/'D' on the graded blocks, 'X' elsewhere."""
    labels = np.full(n_frames, "X", dtype="<U1")
    p_dead = np.asarray(p_dead, dtype=float).copy()
    if pre_kickoff is not None:
        p_dead[pre_kickoff[0]:pre_kickoff[1] + 1] = 1 - cfg.p_floor
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        path = clean_durations(viterbi3(p_home[sl], p_dead[sl], cfg), cfg)
        labels[sl] = np.array(["H", "A", "D"])[path]
    return labels
