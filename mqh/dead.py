"""Dead-ball detection from the tracking alone. Every rule here is a hypothesis, not a calibrated
fact: v0 has not been tuned on the match, and the public 5 minutes are only for sanity checks.

Rules (each can be switched off in Config):
  * pre_kickoff   -- the second half is graded from frame 110808 but kicks off at 113742
                     (FAQ.md); the 2934 frames in between are all dead ball.
  * out_of_bounds -- the ball is tracked outside the pitch rectangle.
  * stationary    -- the ball is tracked but essentially not moving for a couple of seconds.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


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



# ---------------------------------------------------------------------------------------------
# v2: a calibrated dead-ball probability from the tracking (stream 2)
#
# Seven label-free features of the tracking, combined by a logistic model whose weights were fitted
# on EXTERNAL matches only (Metrica's three sample games, degraded to look like our ATD; see
# tools/fit_dead.py, which reproduces them and prints the leave-one-game-out scores). The weights
# are frozen below as constants, so the submission needs no external data at run time.
#
#   walk_ratio   log(players' median speed / its own median over +-5 min): players walk at a stoppage
#   walk_abs     players' median speed (m/s), smoothed over +-2 s
#   walk_long    walk_ratio smoothed over +-5 s (stoppages last several seconds)
#   no_ball      the ball is not tracked on this frame
#   gap_log      log(1 + seconds since the ball was last seen), 0 when it is seen
#   near_line    the ball was last seen within 2 m of a touch / goal line before this gap
#   ball_moving  the ball is seen and moving faster than 3 m/s (a live ball)
#
# Only "a stoppage makes players slow down" and "the ball leaves over a line" are assumed about
# football; no constant refers to a frame of the hackathon match.

FEATURES = ("walk_ratio", "walk_abs", "walk_long", "no_ball", "gap_log", "near_line", "ball_moving")


@dataclass(frozen=True)
class DeadConfig:
    # logistic weights, same order as FEATURES, then the intercept: fitted by tools/fit_dead.py on
    # Metrica's three sample games (degraded), leave-one-game-out AUC 0.92 / 0.92 / 0.92.
    # walk_ratio, no_ball and gap_log are left at 0 on purpose (see tools/fit_dead.py, DROPPED).
    weights: tuple = (0.0, -2.8390, -3.1080, 0.0, 0.0, 0.4940, -2.7410)
    intercept: float = 6.3560
    threshold: float = 0.5           # probability above which a frame is dead
    smooth_frames: int = 25          # +-0.5 s average of the probability before thresholding
    use_pre_kickoff: bool = True
    max_player_speed_ms: float = 12.0   # faster steps are tracking jumps, ignored
    baseline_window_s: float = 600.0    # +-5 min running median of the players' speed
    near_line_m: float = 2.0
    moving_ball_ms: float = 3.0


def dead_features(ball, players, pitch, fps, cfg=DeadConfig()):
    """(n_frames, len(FEATURES)) float array, finite everywhere."""
    n = len(ball)
    scale = np.asarray(pitch, dtype=float)
    p = players * scale
    step = np.full(players.shape[:2], np.nan)
    step[1:] = np.hypot(*(np.moveaxis(p[1:] - p[:-1], -1, 0))) * fps
    step[step > cfg.max_player_speed_ms] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with no tracked player
        med = np.nanmedian(step, axis=1)
    fill = np.nanmedian(med) if np.isfinite(med).any() else 1.0
    med = np.where(np.isfinite(med), med, fill)
    walk_abs = uniform_filter1d(med, 2 * fps + 1)
    sub = max(1, fps)
    coarse = walk_abs[::sub]
    base = median_filter(coarse, size=int(2 * cfg.baseline_window_s) + 1, mode="nearest")
    base = np.repeat(base, sub)[:n]
    walk_ratio = np.log(np.maximum(walk_abs, 0.05) / np.maximum(base, 0.05))
    walk_long = uniform_filter1d(walk_ratio, 10 * fps + 1)

    xy = ball * scale
    missing = np.isnan(ball[:, 0])
    gap_log = np.zeros(n)
    near_line = np.zeros(n)
    starts, ends = _runs(missing)
    for a, b in zip(starts, ends):
        gap_log[a:b] = np.log1p(np.arange(1, b - a + 1) / fps)
        if a > 0:
            x, y = xy[a - 1]
            edge = min(x, scale[0] - x, y, scale[1] - y)
            near_line[a:b] = float(edge < cfg.near_line_m)
    speed = ball_speed_ms(ball, pitch, fps)
    moving = np.nan_to_num(speed, nan=0.0) > cfg.moving_ball_ms
    return np.column_stack([walk_ratio, walk_abs, walk_long, missing.astype(float), gap_log,
                            near_line, moving.astype(float)])


def dead_probability(ball, players, pitch, fps, cfg=DeadConfig()):
    """Per-frame probability that the ball is dead, from the frozen logistic weights."""
    z = dead_features(ball, players, pitch, fps, cfg) @ np.asarray(cfg.weights) + cfg.intercept
    prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    if cfg.smooth_frames > 1:
        prob = uniform_filter1d(prob, cfg.smooth_frames)
    return prob


def tracking_dead_mask(ball, players, pitch, fps, n_frames, pre_kickoff=None, cfg=None):
    """Stream 2, v2: bool (n_frames,), True where the tracking says the ball is dead."""
    cfg = cfg or DeadConfig()
    dead = dead_probability(ball, players, pitch, fps, cfg) >= cfg.threshold
    if cfg.use_pre_kickoff and pre_kickoff is not None:
        dead[pre_kickoff[0]:pre_kickoff[1] + 1] = True
    return dead
