"""Stream 5: turn per-frame evidence into possession labels.

Input per half: a signed evidence score e in [-1, 1] per frame (+ home, - away; 0 = no
information, e.g. the ball is not tracked) and a boolean dead-ball mask. Output: H / A / D labels.

Four ideas, each justified by the structure of the task and not tuned on the match:

0. Clean the dead-ball mask first. "H D H" is two possessions, so one spurious dead frame inside a
   possession splits it in two and one hole inside a stoppage invents a possession. A stoppage
   lasts at least `restart_min_dead_frames` (1 s): shorter dead runs are removed, and live runs
   shorter than `min_live_between_dead_frames` that sit between two dead runs are filled in.

1. Viterbi with a switch penalty. Staying in a state costs -e (home) or +e (away); changing team
   costs `switch_penalty`. A run of L frames can only be supported if its evidence outweighs the
   two switches around it: each frame can swing the score by at most 2, so L > switch_penalty is
   needed. The penalty is therefore a floor on possession length (about 1.2 s of unanimous
   evidence at 25 fps for the default 30).

2. Cheaper switches after a stoppage. A dead-ball stretch ends the previous possession by
   definition, so continuity of the old team is a weaker prior at a restart than in open play.
   The penalty is multiplied by `restart_switch_scale` on the first live frame after a stoppage.

3. Where to put a handover that happens while the evidence is silent. Inside a stretch with no
   informative frames (typically a ball dropout) every position for the switch costs the same,
   except that switching right after a stoppage costs `restart_switch_scale` times less, so the
   restart is the most probable place when one lies in the stretch; otherwise a uniform prior on
   the true time makes the midpoint minimise the expected boundary error, which is what temporal
   IoU punishes. The evidence cost is unchanged, because only silent frames move.

Exactly reproducible: the decoder works in integers (evidence rounded to 1/SCALE), so the same
input gives the same labels on any machine, regardless of floating-point library differences.
No randomness anywhere.
"""

from dataclasses import dataclass

import numpy as np

SCALE = 10_000  # evidence and penalties are rounded to 1/SCALE and decoded as integers


@dataclass(frozen=True)
class DecoderConfig:
    switch_penalty: float = 30.0         # frames' worth of evidence needed to change team
    restart_switch_scale: float = 0.5    # penalty multiplier right after a stoppage
    restart_min_dead_frames: int = 25    # 1 s: a real stoppage lasts at least this long
    place_switch_in_gaps: bool = True    # idea 3
    info_min: float = 0.05               # |evidence| below this carries ~no information
    clean_dead_mask: bool = True         # idea 0
    min_live_between_dead_frames: int = 25  # 1 s: a live burst this short between stoppages is noise


def _runs(mask):
    """(starts, ends_exclusive) of each True run."""
    padded = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    d = np.diff(padded)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def clean_dead_mask(dead, min_dead, min_live_between):
    """Remove dead runs shorter than min_dead, then fill live gaps shorter than min_live_between
    that sit between two dead runs. Never changes the length or crosses the array edges."""
    out = np.asarray(dead, dtype=bool).copy()
    for a, b in zip(*_runs(out)):
        if b - a < min_dead:
            out[a:b] = False
    starts, ends = _runs(out)
    for i in range(len(starts) - 1):
        if starts[i + 1] - ends[i] < min_live_between:
            out[ends[i]:starts[i + 1]] = True
    return out


def _quantise(x):
    return np.rint(np.asarray(x, dtype=float) * SCALE).astype(np.int64)


def viterbi_two_state(evidence, penalty):
    """State path (0 = home, 1 = away) minimising sum(cost) + sum(penalty at each switch).

    `penalty` is a scalar, or one value per step: penalty[t] is the cost of switching between
    step t-1 and step t. Ties prefer staying in the current state. All arithmetic is on integers
    (evidence and penalty rounded to 1/SCALE), so the result does not depend on the platform.
    """
    n = len(evidence)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    ev = _quantise(evidence).tolist()
    pen = _quantise(np.broadcast_to(np.asarray(penalty, dtype=float), (n,))).tolist()
    back_h = np.zeros(n, dtype=bool)  # True: the best path into "home" at t came from away
    back_a = np.zeros(n, dtype=bool)  # True: the best path into "away" at t came from home
    ch, ca = -ev[0], ev[0]            # cost of being home / away so far
    for t in range(1, n):
        from_a = ca + pen[t]
        from_h = ch + pen[t]
        bh = from_a < ch
        ba = from_h < ca
        back_h[t], back_a[t] = bh, ba
        ch, ca = (from_a if bh else ch) - ev[t], (from_h if ba else ca) + ev[t]
    state = 0 if ch <= ca else 1
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        if state == 0 and back_h[t]:
            state = 1
        elif state == 1 and back_a[t]:
            state = 0
    return path


def place_switches(path, informative, restart_before):
    """Move each team change to the best spot inside the silent stretch it falls in.

    path:           state per live frame (0/1)
    informative:    bool per live frame; False where the evidence carries ~no information
    restart_before: bool per live frame; True on the first live frame after a real stoppage

    A change at index s means path[s-1] != path[s]. Let a be the last informative frame before s
    and b the first informative frame at or after s: the frames strictly between them are silent,
    so moving the boundary anywhere in [a+1, b] leaves the evidence cost unchanged.
    """
    out = path.copy()
    n = len(path)
    if n < 2:
        return out
    informative_idx = np.flatnonzero(informative)
    switches = np.flatnonzero(path[1:] != path[:-1]) + 1
    last_hi = 0
    for s in switches:
        pos = np.searchsorted(informative_idx, s)
        if pos == 0 or pos == len(informative_idx):
            continue                       # no informative frame on one side: nothing to anchor to
        a, b = informative_idx[pos - 1], informative_idx[pos]
        lo, hi = a + 1, b
        if lo < last_hi or hi <= lo:
            continue                       # overlaps a previous move, or nothing to move
        anchors = np.flatnonzero(restart_before[lo:hi + 1])
        m = lo + int(anchors[0]) if anchors.size else (lo + hi) // 2
        old, new = path[s - 1], path[s]
        out[lo:m] = old
        out[m:hi] = new
        last_hi = hi
    return out


def decode_half(evidence, dead, first, last, cfg=DecoderConfig()):
    """H / A / D labels for video frames first..last inclusive."""
    sl = slice(first, last + 1)
    ev, dd = evidence[sl], dead[sl]
    if cfg.clean_dead_mask:
        dd = clean_dead_mask(dd, cfg.restart_min_dead_frames, cfg.min_live_between_dead_frames)
    labels = np.full(ev.shape, "D", dtype="<U1")
    live_idx = np.flatnonzero(~dd)
    if live_idx.size == 0:
        return labels

    ev_live = ev[live_idx]
    restart_before = np.zeros(len(live_idx), dtype=bool)
    if len(live_idx) > 1:
        dead_between = np.diff(live_idx) - 1
        restart_before[1:] = dead_between >= cfg.restart_min_dead_frames

    penalty = np.full(len(live_idx), float(cfg.switch_penalty))
    penalty[restart_before] *= cfg.restart_switch_scale

    path = viterbi_two_state(ev_live, penalty)
    if cfg.place_switch_in_gaps:
        informative = np.abs(_quantise(ev_live)) > int(round(cfg.info_min * SCALE))
        path = place_switches(path, informative, restart_before)

    labels[live_idx] = np.where(path == 0, "H", "A")
    return labels
