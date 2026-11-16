"""Turn per-frame evidence into possessions.

Evidence: for every frame where the ball is tracked, a signed score e in [-1, 1] (+ home, - away)
mixing how much closer the nearest home player is than the nearest away player with how many
players of each team surround the ball. It is faded when nobody is near, when the ball is moving
fast (a pass or a shot is not controlled), and when both teams are on the ball (contested). It is
exactly 0 where there is no ball, so a tracking gap neither votes nor flips.

Decoder: a two-state Viterbi (home / away) over the live frames of one half. Staying in a state
costs -e (or +e for the other team); switching costs `switch_penalty`. A switch therefore needs
about `switch_penalty` frames' worth of consistent contrary evidence, which is what turns a
flickering nearest-player vote into possessions. Dead frames are removed first, so
"H D D H" stays two possessions around a stoppage, as the ground truth defines them.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvidenceConfig:
    """Constants of stream 3 (team_evidence). Reasoned, not tuned on this match: docs/DECISIONS.md."""
    margin_scale_m: float = 2.0      # nearest-player distance gap that saturates the vote
    proximity_scale_m: float = 6.0   # how fast the vote fades when nobody is near the ball
    margin_share: float = 0.6        # mix: margin vote vs. nearby-player count vote
    support_radius_m: float = 4.0    # Gaussian radius for "nearby players" counting
    support_eps: float = 0.5         # pseudo-count: a lone far player does not give a full count vote
    speed_mid_ms: float = 10.0       # ball speed where the vote is halved (dribble < mid < pass/shot)
    speed_width_ms: float = 2.0      # softness of that cut
    speed_floor: float = 0.2         # a flying ball keeps a small vote: exact zeros before a stoppage
                                     # let the decoder park its switch there (3-frame fake spells)
    speed_half_window: int = 3       # +-frames for the smoothed ball speed (0.12 s each side at 25 fps)
    contest_radius_m: float = 1.5    # both teams this close to the ball -> contested
    contest_floor: float = 0.35      # vote multiplier for a fully contested ball


# pipeline.Config (stream 5's file) still carries these two original evidence constants; they are
# honoured so its values are never silently ignored. Everything else comes from EvidenceConfig.
_SHARED_FIELDS = ("margin_scale_m", "proximity_scale_m")


def _evidence_cfg(cfg):
    """EvidenceConfig from an EvidenceConfig, None, or the shared pipeline.Config."""
    if isinstance(cfg, EvidenceConfig):
        return cfg
    if cfg is None:
        return EvidenceConfig()
    return EvidenceConfig(**{f: getattr(cfg, f) for f in _SHARED_FIELDS if hasattr(cfg, f)})


def smoothed_ball_speed(ball, pitch, fps=25, half_window=3):
    """Ball speed (m/s) from a centred difference over up to +-half_window frames.

    A one-frame difference turns ~0.3 m of tracking jitter into ~10 m/s, which would make every
    dribble look like a pass. Uses the widest symmetric window with both ends tracked, then falls
    back to narrower ones; NaN where the ball is untracked or isolated.
    """
    xy = ball * np.asarray(pitch, dtype=float)
    n = len(xy)
    speed = np.full(n, np.nan)
    for k in range(max(1, half_window), 0, -1):
        if n <= 2 * k:
            continue
        dx = xy[2 * k:] - xy[:-2 * k]
        v = np.hypot(dx[:, 0], dx[:, 1]) * fps / (2 * k)
        target = speed[k:n - k]
        fill = np.isnan(target) & ~np.isnan(v) & ~np.isnan(xy[k:n - k, 0])
        target[fill] = v[fill]
    # ends of tracked runs: one-sided difference
    if n > 1:
        step = np.hypot(*(xy[1:] - xy[:-1]).T) * fps
        fwd = np.isnan(speed[:-1]) & ~np.isnan(step)
        speed[:-1][fwd] = step[fwd]
        bwd = np.isnan(speed[1:]) & ~np.isnan(step)
        speed[1:][bwd] = step[bwd]
    return speed


def team_evidence(ball, players, team_of_slot, pitch, cfg, fps=25, return_parts=False):
    """Signed per-frame evidence in [-1, 1] (+ home, - away); 0 where there is no usable ball.

    evidence = vote * w_near * w_speed * w_contest, where
      vote      = margin_share * tanh((d_away - d_home) / margin_scale)
                  + (1 - margin_share) * (S_home - S_away) / (S_home + S_away + eps)
                  with d_* the nearest player of each team and S_* = sum_i exp(-d_i^2 / 2 r^2)
                  a soft count of each team's players around the ball;
      w_near    = exp(-nearest / proximity_scale)        nobody near -> nobody controls it;
      w_speed   = floor + (1 - floor) / (1 + exp((v - v_mid) / v_width))
                                                         a flying ball is (mostly) not controlled;
      w_contest = 1 - (1 - floor) * c,  c = k(d_home) k(d_away), k(d) = exp(-(d / r_c)^2)
                                                         both teams on the ball -> weaker vote.
    With return_parts=True also returns a dict of the per-frame components, for debugging.
    """
    c = _evidence_cfg(cfg)
    scale = np.asarray(pitch, dtype=float)
    n = len(ball)
    evidence = np.zeros(n)
    parts = {name: np.full(n, np.nan) for name in
             ("d_home", "d_away", "s_home", "s_away", "vote", "w_near", "speed", "w_speed",
              "w_contest")}
    idx = np.where(~np.isnan(ball[:, 0]))[0]
    if idx.size == 0:
        return (evidence, parts) if return_parts else evidence

    d = np.hypot(*(np.moveaxis(players[idx] * scale - (ball[idx] * scale)[:, None, :], -1, 0)))
    d = np.where(np.isnan(d), np.inf, d)
    home = team_of_slot == 0
    inf = np.full(len(idx), np.inf)
    d_home = d[:, home].min(axis=1) if home.any() else inf
    d_away = d[:, ~home].min(axis=1) if (~home).any() else inf

    # nearest-player margin: only meaningful when both teams are tracked
    both = np.isfinite(d_home) & np.isfinite(d_away)
    margin = np.zeros(len(idx))
    margin[both] = d_away[both] - d_home[both]
    margin_vote = np.tanh(margin / c.margin_scale_m)

    # soft count of nearby players per team (exp(-inf) = 0 for untracked slots)
    g = np.exp(-0.5 * (d / c.support_radius_m) ** 2)
    s_home = g[:, home].sum(axis=1)
    s_away = g[:, ~home].sum(axis=1)
    support_vote = (s_home - s_away) / (s_home + s_away + c.support_eps)

    a = c.margin_share
    vote = a * margin_vote + (1 - a) * support_vote

    nearest = np.minimum(d_home, d_away)
    w_near = np.where(np.isfinite(nearest), np.exp(-nearest / c.proximity_scale_m), 0.0)

    speed = smoothed_ball_speed(ball, pitch, fps, int(c.speed_half_window))[idx]
    z = np.clip((speed - c.speed_mid_ms) / c.speed_width_ms, -50, 50)
    f = c.speed_floor
    w_speed = np.where(np.isnan(speed), 1.0, f + (1.0 - f) / (1.0 + np.exp(z)))

    r_c = c.contest_radius_m
    contest = np.exp(-(d_home / r_c) ** 2) * np.exp(-(d_away / r_c) ** 2)
    w_contest = 1.0 - (1.0 - c.contest_floor) * contest

    evidence[idx] = np.clip(vote * w_near * w_speed * w_contest, -1.0, 1.0)
    if not return_parts:
        return evidence
    for name, val in (("d_home", d_home), ("d_away", d_away), ("s_home", s_home),
                      ("s_away", s_away), ("vote", vote), ("w_near", w_near), ("speed", speed),
                      ("w_speed", w_speed), ("w_contest", w_contest)):
        parts[name][idx] = val
    return evidence, parts


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
