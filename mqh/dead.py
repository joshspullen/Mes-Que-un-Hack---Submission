"""Dead-ball detection from the tracking alone. Every rule here is a hypothesis, not a calibrated
fact: v0 has not been tuned on the match, and the public 5 minutes are only for sanity checks.

Rules (each can be switched off in Config):
  * pre_kickoff   -- the second half is graded from frame 110808 but kicks off at 113742
                     (FAQ.md); the 2934 frames in between are all dead ball.
  * out_of_bounds -- the ball is tracked outside the pitch rectangle.
  * stationary    -- the ball is tracked but essentially not moving for a couple of seconds.
"""

import numpy as np


def ball_speed_ms(ball, pitch, fps, max_gap=5):
    """Ball speed in m/s from consecutive tracked positions; NaN across long gaps."""
    length, width = pitch
    xy = ball * np.array([length, width])
    seen = ~np.isnan(xy[:, 0])
    idx = np.where(seen)[0]
    speed = np.full(len(ball), np.nan)
    if len(idx) < 2:
        return speed
    step = np.hypot(np.diff(xy[idx, 0]), np.diff(xy[idx, 1]))
    gap = np.diff(idx)
    ok = gap <= max_gap
    speed[idx[1:][ok]] = step[ok] / gap[ok] * fps
    return speed


def _runs(mask):
    """(start, end_exclusive) of each True run."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(padded)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def _close(mask, max_gap):
    """Fill False gaps of <= max_gap frames that sit between True frames."""
    out = mask.copy()
    s, e = _runs(mask)
    for i in range(len(s) - 1):
        if s[i + 1] - e[i] <= max_gap:
            out[e[i]:s[i + 1]] = True
    return out


def dead_mask(ball, speed, cfg, n_frames, pre_kickoff=None):
    """Boolean per-frame 'ball is dead' over the whole video."""
    dead = np.zeros(n_frames, dtype=bool)

    if cfg.use_out_of_bounds:
        x, y = ball[:, 0], ball[:, 1]
        m = cfg.oob_margin
        with np.errstate(invalid="ignore"):
            oob = (x < -m) | (x > 1 + m) | (y < -m) | (y > 1 + m)
        oob = _close(oob, cfg.oob_close_frames)
        # the ball stays dead for a moment after it comes back into view
        s, e = _runs(oob)
        for a, b in zip(s, e):
            oob[b:b + cfg.oob_tail_frames] = True
        dead |= oob

    if cfg.use_stationary:
        slow = np.zeros(n_frames, dtype=bool)
        with np.errstate(invalid="ignore"):
            slow[~np.isnan(speed)] = speed[~np.isnan(speed)] < cfg.still_speed_ms
        slow = _close(slow, cfg.still_close_frames)
        s, e = _runs(slow)
        for a, b in zip(s, e):
            if b - a >= cfg.still_min_frames:
                dead[a:b] = True

    if cfg.use_pre_kickoff and pre_kickoff is not None:
        dead[pre_kickoff[0]:pre_kickoff[1] + 1] = True
    return dead
