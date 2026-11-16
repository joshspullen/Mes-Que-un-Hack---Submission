"""Make a clean external match look like our automatic tracking (BAR-ATH ATD).

What the ATD is known to look like (DATA.md, label-free): the ball is missing on 41% of graded
frames, 20.5 players are tracked on average instead of 22, positions are noisy, and the ball tends
to vanish in crowded or aerial moments. Metrica's sample games already hide the ball during almost
every stoppage, so only live play needs extra gaps.

Every number below is an ASSUMPTION about BAR-ATH, not a measurement of its labels. The defaults
put live-play ball dropouts at 25% of live frames; `tools/sim_check.py` prints the same label-free
statistics for BAR-ATH and for the degraded games so the team can check them on the real tracking
and change the defaults here if they disagree. Deterministic for a given seed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

PITCH = np.array([105.0, 68.0])


@dataclass(frozen=True)
class DegradeConfig:
    live_ball_missing: float = 0.25      # target share of live frames with no ball (incl. existing gaps)
    gap_median_s: float = 0.8            # lognormal median length of an added ball gap
    gap_sigma: float = 1.0               # lognormal spread (long tail of multi-second losses)
    crowd_radius_m: float = 3.0          # a ball with >= crowd_min players this close is easier to lose
    crowd_min: int = 3
    crowd_weight: float = 3.0            # relative chance that an added gap starts in a crowd
    ball_noise_m: float = 0.25
    player_noise_m: float = 0.10
    players_tracked: float = 20.5        # mean tracked players per frame on BAR-ATH (DATA.md)
    dropout_mean_s: float = 2.0          # mean length of a player dropout
    ball_offset_m: float = 0.0           # slowly varying error of the ball position (std, metres)
    offset_span_s: float = 1.0           # time scale of that error
    stoppage_visible_share: float = 0.0  # share of stoppages where the ball stays visible a while
    stoppage_visible_s: tuple = (1.0, 6.0)   # how long it stays visible (uniform)
    fill_short_gaps_s: float = 0.0       # interpolate the source's own ball gaps up to this long
    dead_ball_missing: float = 0.0       # share of stoppages where a source that keeps tracking
                                         # the dead ball (TRACAB) loses it after 1-6 s
    fps: int = 25


# A second profile, closer to our match. Its numbers come from a label-free calibration published
# by another team (MikolajEkiert/more-than-a-hack-2027, docs/DECISIONS.md, 14 Nov, profile
# `match_like`): fitted so that agreement statistics between two pipelines on the real tracking
# (no labels) are reproduced on degraded external games. Best of their grid: 25% of live frames
# lose the ball, ball offset 2.0 m, the ball stays visible after 90% of stoppages for 1-6 s.
MATCH_LIKE = DegradeConfig(ball_offset_m=2.0, stoppage_visible_share=0.9, dead_ball_missing=0.35)
# A third profile from label-free statistics of BAR-ATH's own tracking (tools/sim_check.py, 15 Nov):
# the ball is missing on 40.7% of graded frames, in few long gaps (1.4 per minute, median 10.7 s,
# 90th percentile 34 s), no more often inside set-piece tags (32%) than elsewhere (44%), and player
# positions are very smooth (second difference 0.001 m). Ball offset and stoppage visibility as in
# MATCH_LIKE. tools/sim_check.py prints the same statistics for the degraded games.
BAR_ATH = DegradeConfig(live_ball_missing=0.30, gap_median_s=10.0, gap_sigma=0.8, crowd_weight=1.0,
                        ball_noise_m=0.05, player_noise_m=0.0, ball_offset_m=2.0,
                        stoppage_visible_share=0.9, fill_short_gaps_s=1.0, dead_ball_missing=0.35)
PROFILES = {"default": DegradeConfig(), "match_like": MATCH_LIKE, "bar_ath": BAR_ATH}


def _runs(mask):
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(padded)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def degrade(ball, players, live, seed=0, cfg=DegradeConfig()):
    """(ball, players) degraded copies. `live` = bool per frame, where extra ball gaps may go."""
    rng = np.random.default_rng(seed)
    n = len(ball)
    ball = ball.copy()
    players = players.copy()

    if cfg.fill_short_gaps_s > 0:   # our tracking interpolates short losses; Metrica's has many
        miss = np.isnan(ball[:, 0])
        starts, ends = _runs(miss)
        idx = np.arange(n)
        ok = ~miss
        if ok.sum() > 1:
            fx = np.interp(idx, idx[ok], ball[ok, 0])
            fy = np.interp(idx, idx[ok], ball[ok, 1])
            limit = cfg.fill_short_gaps_s * cfg.fps
            for a, b in zip(starts, ends):
                if a > 0 and b < n and b - a <= limit:
                    ball[a:b, 0], ball[a:b, 1] = fx[a:b], fy[a:b]

    # the ball stays in view for a while after many stoppages (Metrica's sample data hides it at
    # once, our automatic tracking often does not): hold its last position, drifting slowly
    if cfg.stoppage_visible_share > 0:
        dead = ~live
        starts = np.flatnonzero(dead & np.concatenate(([False], ~dead[:-1])))
        for a in starts:
            if rng.random() >= cfg.stoppage_visible_share:
                continue
            seen = np.flatnonzero(~np.isnan(ball[max(0, a - 2 * cfg.fps):a + 1, 0]))
            if seen.size == 0:
                continue
            last = ball[max(0, a - 2 * cfg.fps) + seen[-1]].copy()
            length = int(rng.uniform(*cfg.stoppage_visible_s) * cfg.fps)
            end = a + length
            while end > a and not dead[min(end, n) - 1]:
                end -= 1
            span = slice(a, min(n, end))
            drift = np.cumsum(rng.normal(0, 0.02, size=(span.stop - span.start, 2)), axis=0) / PITCH
            ball[span] = last + drift

    # position noise, smooth over ~0.2 s so speeds stay physical
    for arr, sd in ((ball, cfg.ball_noise_m), (players, cfg.player_noise_m)):
        noise = uniform_filter1d(rng.normal(0.0, sd, size=arr.shape), 5, axis=0) * np.sqrt(5)
        arr += noise / PITCH
    if cfg.ball_offset_m > 0:
        span = max(1, int(cfg.offset_span_s * cfg.fps))
        off = uniform_filter1d(rng.normal(0.0, 1.0, size=ball.shape), span, axis=0)
        off *= cfg.ball_offset_m / max(1e-9, off.std())
        ball += off / PITCH

    # a provider that tracks the dead ball (TRACAB) still loses it at some stoppages in ours
    if cfg.dead_ball_missing > 0:
        dead = ~live
        starts, ends = _runs(dead)
        for a, b in zip(starts, ends):
            if rng.random() < cfg.dead_ball_missing:
                ball[min(b, a + int(rng.uniform(1.0, 6.0) * cfg.fps)):b] = np.nan

    # extra ball gaps in live play, preferring crowded moments
    missing = np.isnan(ball[:, 0])
    target = int(cfg.live_ball_missing * live.sum())
    have = int((missing & live).sum())
    if target > have:
        xy = ball * PITCH
        d = np.hypot(*(np.moveaxis(players * PITCH - xy[:, None, :], -1, 0)))
        crowd = (np.nan_to_num(d, nan=1e9) < cfg.crowd_radius_m).sum(axis=1) >= cfg.crowd_min
        w = np.where(live & ~missing, np.where(crowd, cfg.crowd_weight, 1.0), 0.0)
        cand = np.flatnonzero(w > 0)
        p = w[cand] / w[cand].sum()
        added = 0
        while added < target - have and cand.size:
            start = int(rng.choice(cand, p=p))
            length = max(1, int(round(rng.lognormal(np.log(cfg.gap_median_s * cfg.fps), cfg.gap_sigma))))
            seg = slice(start, min(n, start + length))
            new = live[seg] & ~missing[seg]
            missing[seg] |= live[seg]
            added += int(new.sum())
        ball[missing] = np.nan

    # player dropouts down to the BAR-ATH average
    seen = ~np.isnan(players[:, :, 0])
    on_pitch = seen.sum(axis=1).mean()
    if on_pitch > cfg.players_tracked:
        drop_share = 1.0 - cfg.players_tracked / on_pitch
        mean_len = cfg.dropout_mean_s * cfg.fps
        start_p = drop_share / mean_len / max(1e-9, 1.0 - drop_share)
        for j in range(players.shape[1]):
            starts = np.flatnonzero(rng.random(n) < start_p)
            for s in starts:
                players[s:s + int(rng.exponential(mean_len)) + 1, j] = np.nan
    return ball, players
