"""Stream 5: turn per-frame evidence into possession labels.

Input per half: a signed evidence score e in [-1, 1] per frame (+ home, - away; 0 = no
information, e.g. the ball is not tracked) and a boolean dead-ball mask. Output: H / A / D labels.

Three ideas, each justified by the structure of the task and not tuned on the match:

1. Viterbi with a switch penalty. Staying in a state costs -e (home) or +e (away); changing team
   costs `switch_penalty`. A run of L frames can only be supported if its evidence outweighs the
   two switches around it: each frame can swing the score by at most 2, so L > switch_penalty is
   needed. The penalty is therefore a floor on possession length (about 1.2 s of unanimous
   evidence at 25 fps for the default 30).

2. Cheaper switches after a stoppage. A dead-ball stretch ends the previous possession by
   definition ("H D H" is two possessions), so continuity of the old team is a weaker prior at a
   restart than in open play. The penalty is multiplied by `restart_switch_scale` on the first live
   frame after a dead stretch of at least `restart_min_dead_frames`.

3. Where to put a handover that happens while the evidence is silent. Inside a stretch with no
   informative frames (typically a ball dropout) every position for the switch costs the same, so
   the Viterbi tie-break puts it arbitrarily. We place it where the error is smallest in
   expectation: at the restart if a dead stretch lies in the stretch (that is when a new possession
   starts), otherwise at the midpoint (a uniform prior on the true time minimises the expected
   boundary error, which is what temporal IoU punishes). The cost is unchanged, because only
   frames with ~zero evidence move.

No randomness anywhere: the same input gives the same labels.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    switch_penalty: float = 30.0         # frames' worth of evidence needed to change team
    restart_switch_scale: float = 0.5    # penalty multiplier right after a stoppage
    restart_min_dead_frames: int = 25    # 1 s: shorter dead blips are not a real stoppage
    place_switch_in_gaps: bool = True    # idea 3
    info_min: float = 0.05               # |evidence| below this carries ~no information


def viterbi_two_state(evidence, penalty):
    """State path (0 = home, 1 = away) minimising sum(cost) + sum(penalty at each switch).

    `penalty` is a scalar, or one value per step: penalty[t] is the cost of switching between
    step t-1 and step t. Ties prefer staying in the current state.
    """
    n = len(evidence)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    pen = np.broadcast_to(np.asarray(penalty, dtype=float), (n,))
    cost_home = -evidence
    cost_away = evidence
    back_h = np.zeros(n, dtype=bool)  # True: the best path into "home" at t came from away
    back_a = np.zeros(n, dtype=bool)  # True: the best path into "away" at t came from home
    ch, ca = cost_home[0], cost_away[0]
    for t in range(1, n):
        p = pen[t]
        from_a = ca + p
        from_h = ch + p
        back_h[t] = from_a < ch
        back_a[t] = from_h < ca
        nh = min(ch, from_a) + cost_home[t]
        na = min(ca, from_h) + cost_away[t]
        ch, ca = nh, na
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
        path = place_switches(path, np.abs(ev_live) > cfg.info_min, restart_before)

    labels[live_idx] = np.where(path == 0, "H", "A")
    return labels
