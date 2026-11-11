"""Turn per-frame evidence into possessions.

Evidence: for every frame where the ball is tracked, how much closer is the nearest home player
than the nearest away player, weighted by how close anybody is. It is a signed score e in [-1, 1]
(+ home, - away) and exactly 0 where there is no ball, so a tracking gap neither votes nor flips.

Decoder: a two-state Viterbi (home / away) over the live frames of one half. Staying in a state
costs -e (or +e for the other team); switching costs `switch_penalty`. A switch therefore needs
about `switch_penalty` frames' worth of consistent contrary evidence, which is what turns a
flickering nearest-player vote into possessions. Dead frames are removed first, so
"H D D H" stays two possessions around a stoppage, as the ground truth defines them.
"""

import numpy as np


def team_evidence(ball, players, team_of_slot, pitch, cfg):
    """Signed per-frame evidence in [-1, 1]; 0 where there is no usable ball/players."""
    length, width = pitch
    scale = np.array([length, width])
    n = len(ball)
    evidence = np.zeros(n)
    have_ball = ~np.isnan(ball[:, 0])
    idx = np.where(have_ball)[0]
    if idx.size == 0:
        return evidence

    d = np.hypot(*(np.moveaxis(players[idx] * scale - (ball[idx] * scale)[:, None, :], -1, 0)))
    d = np.where(np.isnan(d), np.inf, d)
    home = team_of_slot == 0
    d_home = d[:, home].min(axis=1) if home.any() else np.full(len(idx), np.inf)
    d_away = d[:, ~home].min(axis=1) if (~home).any() else np.full(len(idx), np.inf)

    both = np.isfinite(d_home) & np.isfinite(d_away)
    margin = np.zeros(len(idx))
    margin[both] = d_away[both] - d_home[both]
    nearest = np.minimum(d_home, d_away)
    weight = np.zeros(len(idx))
    weight[both] = np.exp(-nearest[both] / cfg.proximity_scale_m)
    evidence[idx] = np.tanh(margin / cfg.margin_scale_m) * weight
    return evidence


def viterbi_two_state(evidence, switch_penalty):
    """State path (0 = home, 1 = away) minimising  sum(cost) + switch_penalty * #switches."""
    n = len(evidence)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    cost_home = -evidence
    cost_away = evidence
    back_h = np.zeros(n, dtype=bool)  # True: came from away
    back_a = np.zeros(n, dtype=bool)  # True: came from home
    ch, ca = cost_home[0], cost_away[0]
    for t in range(1, n):
        from_a = ca + switch_penalty
        from_h = ch + switch_penalty
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


def decode_half(evidence, dead, first, last, cfg):
    """Labels ('H'/'A'/'D') for video frames first..last inclusive."""
    sl = slice(first, last + 1)
    ev, dd = evidence[sl], dead[sl]
    labels = np.full(ev.shape, "D", dtype="<U1")
    live = ~dd
    path = viterbi_two_state(ev[live], cfg.switch_penalty)
    labels[live] = np.where(path == 0, "H", "A")
    return labels
