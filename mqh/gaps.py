"""Stream 4: who has the ball while the ball is not tracked, from how the two teams move.

    gap_evidence(ball, players, team_of_slot, pitch, fps) -> float (n_frames,) in [-1, 1]

The pipeline uses it only where the ball is missing (41% of graded frames on BAR-ATH). Football
reasoning: the team in possession spreads out to make the pitch big, the team without it closes
up; the team in possession moves towards the opponent's goal. Three antisymmetric features
(home minus away), so swapping the teams flips the sign and nothing favours either side:

  width   difference of the teams' spread across the pitch (std of y), +-1 s average
  depth   difference of the teams' spread along the pitch (std of x), +-1 s average
  push    difference of the teams' centroid speed towards the opponent's goal, +-1 s average

Attack direction is read from the tracking itself: whichever team's centroid sits nearer a goal,
over a centred 20-minute window, defends that goal (teams change ends at half time).

The weights are fitted on EXTERNAL matches only (Metrica's sample games, degraded; logistic with
no intercept, see tools/fit_gaps.py): leave-one-game-out AUC for "home has the ball" is 0.83-0.87
over all live frames and 0.61-0.67 inside ball gaps, where possession changes are concentrated.
`weight` scales the vote so a shape guess never outweighs a seen ball (tools/external_bench.py).
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class GapConfig:
    coef: tuple = (0.3295, 0.0430, 0.1314)   # width, depth, push (tools/fit_gaps.py)
    weight: float = 0.5                      # scale of the vote in [-1, 1]
    smooth_s: float = 2.0
    direction_window_s: float = 1200.0


def shape_features(players, team_of_slot, pitch, fps, cfg=GapConfig()):
    p = players * np.asarray(pitch, dtype=float)
    team_of_slot = np.asarray(team_of_slot)
    win = max(1, int(cfg.smooth_s * fps) | 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with nobody tracked
        stats = []
        for t in (0, 1):
            q = p[:, team_of_slot == t]
            stats.append((np.nanmean(q[:, :, 0], axis=1), np.nanstd(q[:, :, 0], axis=1),
                          np.nanstd(q[:, :, 1], axis=1)))
    (cxh, sxh, syh), (cxa, sxa, sya) = stats

    def fill(v):
        v = np.asarray(v, dtype=float)
        ok = np.isfinite(v)
        if not ok.any():
            return np.zeros_like(v)
        idx = np.arange(len(v))
        return np.interp(idx, idx[ok], v[ok])

    cxh, cxa, sxh, sxa, syh, sya = map(fill, (cxh, cxa, sxh, sxa, syh, sya))
    long_win = max(1, int(cfg.direction_window_s * fps) | 1)
    # +1 where home attacks towards increasing x (its centroid sits on the low-x side)
    direction = np.sign(uniform_filter1d(cxa - cxh, long_win, mode="nearest"))
    direction[direction == 0] = 1.0
    vh = np.gradient(cxh) * fps
    va = np.gradient(cxa) * fps
    width = uniform_filter1d(syh - sya, win)
    depth = uniform_filter1d(sxh - sxa, win)
    # home's speed towards the away goal minus away's speed towards the home goal
    push = uniform_filter1d((vh * direction) - (va * -direction), win)
    return np.column_stack([width, depth, push])


def gap_evidence(ball, players, team_of_slot, pitch, fps, cfg=GapConfig()):
    z = shape_features(players, team_of_slot, pitch, fps, cfg) @ np.asarray(cfg.coef)
    return np.clip(cfg.weight * np.tanh(z / 2.0), -1.0, 1.0)
