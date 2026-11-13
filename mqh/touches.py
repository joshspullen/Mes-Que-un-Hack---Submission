"""Stream 3b: ball touches from the tracking (docs/INTERFACES.md: touch_events).

A touch is a sudden change of the ball's velocity with a player close to it: a pass leaving a foot,
a reception, a header, a tackle. The team of the nearest player is the team that touched it.

    touch_events(ball, players, team_of_slot, pitch, fps) -> (frames, teams)

Used by stream 5 to snap each possession change onto the new team's first touch (mqh/refine.py),
because temporal IoU punishes late boundaries hard (8 frames late costs about 0.11 spellF1).

Constants are physical, not fitted on the hackathon match. tools/check_touches.py measures them
against Metrica's sample events (external): precision / recall of touches against pass starts and
receptions.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d


@dataclass(frozen=True)
class TouchConfig:
    half_window: int = 3          # velocity before / after over +-3 frames (0.12 s at 25 fps)
    min_dv_ms: float = 6.0        # velocity change that counts as a touch (tools/check_touches.py)
    max_player_m: float = 2.0     # someone must be this close to the ball
    min_gap_frames: int = 5       # two touches closer than 0.2 s are one touch
    max_ball_jump_ms: float = 45.0   # faster than any kick: a tracking jump, not a touch


def _velocity(xy, fps, k):
    """Mean velocity over the k frames before and after each frame; NaN if the window has a gap."""
    n = len(xy)
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[k:] = (xy[k:] - xy[:-k]) * fps / k
    after[:-k] = (xy[k:] - xy[:-k]) * fps / k
    return before, after


def touch_events(ball, players, team_of_slot, pitch, fps, cfg=TouchConfig()):
    scale = np.asarray(pitch, dtype=float)
    xy = ball * scale
    k = cfg.half_window
    before, after = _velocity(xy, fps, k)
    dv = np.hypot(*(after - before).T)
    sp_b, sp_a = np.hypot(*before.T), np.hypot(*after.T)
    valid = np.isfinite(dv) & (sp_b < cfg.max_ball_jump_ms) & (sp_a < cfg.max_ball_jump_ms)
    dv = np.where(valid, dv, 0.0)

    peak = (dv >= cfg.min_dv_ms) & (dv == maximum_filter1d(dv, 2 * cfg.min_gap_frames + 1))
    idx = np.flatnonzero(peak)
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    d = np.hypot(*np.moveaxis(players[idx] * scale - xy[idx][:, None, :], -1, 0))
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.argmin(axis=1)
    close = d[np.arange(len(idx)), nearest] <= cfg.max_player_m
    frames = idx[close].astype(np.int64)
    teams = np.asarray(team_of_slot)[nearest[close]].astype(np.int64)
    return frames, teams
