"""Stream 2b: learned dead-ball probability -- is play stopped, from many weak signals at once.

Stream 2 v2 (mqh/dead.py) is a 4-feature logistic model fitted on 3 Metrica games. With 30 external
matches from three providers (mqh/external/registry.py), including the DFL's own per-frame ball
status (IDSSE), a gradient-boosted model can read stoppages from the shape of the game:

  - players walking (median speed, share of players almost still), against their own baseline
  - the ball last seen near a line, the ball not moving, how long it has been unseen
  - restart set-ups: players gathered in a penalty area (corner, free kick near the box), a team
    split into its own half with a player at the centre spot (kick-off), compactness of all players
  - all of it with its context, +-1 s, +-3 s and +-10 s, and before / after (a stoppage starts
    with a sudden slowdown and ends with a burst)

Everything is symmetric between the teams (dead ball has no side). Trained at run time, like the
team model, single-threaded and seeded (same file twice).
"""

import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from mqh.dead import DeadConfig, dead_features

HGB_PARAMS = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                  early_stopping=False, random_state=0)
SUBSAMPLE = 5


def _speeds(players, pitch, fps, span_s=1.0):
    p = players * np.asarray(pitch, dtype=float)
    k = max(1, int(round(span_s * fps / 2)))
    sp = np.full(players.shape[:2], np.nan)
    sp[k:-k] = np.hypot(*(np.moveaxis(p[2 * k:] - p[:-2 * k], -1, 0))) * fps / (2 * k)
    sp[sp > 12.0] = np.nan
    return sp


def features(ball, players, team_of_slot, pitch, fps):
    """(n, F) float32, finite: the stream-2 features plus shape, set-up and context features."""
    scale = np.asarray(pitch, dtype=float)
    team = np.asarray(team_of_slot)
    n = len(ball)
    base = dead_features(ball, players, pitch, fps, DeadConfig())          # 7 columns
    p = players * scale
    xy = ball * scale
    sp = _speeds(players, pitch, fps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        still = np.nanmean(sp < 1.0, axis=1)
        fast = np.nanmean(sp > 4.0, axis=1)
        tracked = (~np.isnan(p[:, :, 0])).sum(axis=1).astype(float)
        sx = np.nanstd(p[:, :, 0], axis=1)
        sy = np.nanstd(p[:, :, 1], axis=1)
        cx = np.nanmean(p[:, :, 0], axis=1)
        # players in either penalty area (16.5 m deep, 40.3 m wide)
        in_y = np.abs(p[:, :, 1] - scale[1] / 2) < 20.15
        box_l = np.nansum((p[:, :, 0] < 16.5) & in_y, axis=1)
        box_r = np.nansum((p[:, :, 0] > scale[0] - 16.5) & in_y, axis=1)
        crowd_box = np.maximum(box_l, box_r).astype(float)
        # kick-off set-up: each team in one half, someone near the centre spot
        cxh = np.nanmean(p[:, team == 0, 0], axis=1)
        cxa = np.nanmean(p[:, team == 1, 0], axis=1)
        side_h = np.sign(np.nan_to_num(cxa - cxh))
        own = np.where(side_h[:, None] > 0,
                       np.where(team[None, :] == 0, p[:, :, 0] <= scale[0] / 2 + 1, p[:, :, 0] >= scale[0] / 2 - 1),
                       np.where(team[None, :] == 0, p[:, :, 0] >= scale[0] / 2 - 1, p[:, :, 0] <= scale[0] / 2 + 1))
        own_share = np.nanmean(np.where(np.isnan(p[:, :, 0]), np.nan, own), axis=1)
        centre = np.nanmin(np.hypot(p[:, :, 0] - scale[0] / 2, p[:, :, 1] - scale[1] / 2), axis=1)
    seen = ~np.isnan(ball[:, 0])
    # last seen ball position, carried through gaps
    last = np.where(seen[:, None], xy, np.nan)
    idx = np.where(seen, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    last = xy[idx]
    last = np.where(np.isnan(last), scale / 2, last)
    edge = np.minimum.reduce([last[:, 0], scale[0] - last[:, 0], last[:, 1], scale[1] - last[:, 1]])
    to_centre = np.hypot(last[:, 0] - scale[0] / 2, last[:, 1] - scale[1] / 2)
    out_pitch = ((last[:, 0] < 0) | (last[:, 0] > scale[0]) | (last[:, 1] < 0) | (last[:, 1] > scale[1])).astype(float)

    def fill(v, default):
        v = np.asarray(v, dtype=float)
        return np.where(np.isfinite(v), v, default)

    cols = [base,
            np.column_stack([fill(still, 0.5), fill(fast, 0.1), tracked, fill(sx, 20), fill(sy, 10),
                             fill(cx, scale[0] / 2) - scale[0] / 2, crowd_box, fill(own_share, 0.5),
                             fill(centre, 30), np.minimum(edge, 40), to_centre, out_pitch])]
    ctx = [fill(still, 0.5), base[:, 1], fill(fast, 0.1), crowd_box]
    for w_s in (1, 3, 10):
        w = int(w_s * fps) | 1
        cols.append(np.column_stack([uniform_filter1d(c, w) for c in ctx]))
    for w_s in (3, 10):   # what comes before and after: a stoppage starts slow, ends with a burst
        w = int(w_s * fps)
        prev = np.column_stack([np.concatenate([np.full(w, c[0]), uniform_filter1d(c, w)[:-w]]) for c in ctx[:3]])
        nxt = np.column_stack([np.concatenate([uniform_filter1d(c, w)[w:], np.full(w, c[-1])]) for c in ctx[:3]])
        cols += [prev, nxt]
    cols.append(np.column_stack([maximum_filter1d(base[:, 6], int(3 * fps) | 1),
                                 minimum_filter1d(fill(still, 0.5), int(3 * fps) | 1)]))
    return np.hstack(cols).astype(np.float32)


def fit(rows):
    """rows: [(features, truth)] -> model of P(dead). Graded frames only."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    X, y = [], []
    for F, truth in rows:
        k = np.flatnonzero(truth != "X")[::SUBSAMPLE]
        X.append(F[k])
        y.append(truth[k] == "D")
    with threadpool_limits(1):
        return HistGradientBoostingClassifier(**HGB_PARAMS).fit(np.vstack(X), np.concatenate(y))


def probability(model, F):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        return model.predict_proba(F)[:, 1]
