"""Self-training (stream 3d): adapt the team and dead-ball models to the target match's own tracking.

Round 0 trains on external matches only. Each further round adds the target frames the models are
very sure about, labelled by the models themselves (pseudo-labels produced by an algorithm, never
by a person: allowed by RULES.md), with a reduced weight, and retrains. Frames near a predicted
change are left out (that is where the models are least reliable and where the score is decided).

Whether it helps is measured on external matches by tools/check_self_training.py before it is
used; see docs/DECISIONS.md for the outcome.
"""

import numpy as np

from mqh import dead_model, team_model

CONF_TEAM = 0.85        # |evidence| above this: a confident team label
CONF_DEAD = 0.9         # p_dead above this (dead) or below 1 - this (alive)
MARGIN_S = 2.0          # keep away from predicted changes
WEIGHT = 0.5            # weight of a pseudo-labelled frame against an external one
SUBSAMPLE = 5


def _far_from_changes(labels, fps=25, margin_s=MARGIN_S):
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    far = np.ones(len(labels), dtype=bool)
    m = int(margin_s * fps)
    for c in change:
        far[max(0, c - m):c + m] = False
    return far


def fit_predict(team_rows, dead_rows, target, graded, rounds=1):
    """(team evidence, p_dead) for the target after `rounds` of self-training.

    team_rows / dead_rows: external training rows, already subsampled (anti, sym, truth) / (F, truth).
    target: dict(anti, sym, deadF) full-length arrays of the target match. graded: bool mask.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    Xt = np.vstack([np.hstack([a, s]) for a, s, _ in team_rows] + [np.hstack([-a, s]) for a, s, _ in team_rows])
    yt = np.concatenate([t == "H" for _, _, t in team_rows] + [t != "H" for _, _, t in team_rows])
    live_mask = [np.isin(t, ["H", "A"]) for _, _, t in team_rows]
    keep_t = np.concatenate(live_mask + live_mask)
    Xt, yt = Xt[keep_t], yt[keep_t]
    Xd = np.vstack([F for F, _ in dead_rows])
    yd = np.concatenate([t == "D" for _, t in dead_rows])
    keep_d = np.concatenate([t != "X" for _, t in dead_rows])
    Xd, yd = Xd[keep_d], yd[keep_d]
    wt, wd = np.ones(len(yt)), np.ones(len(yd))

    anti, sym, F = target["anti"], target["sym"], target["deadF"]
    ev = pd = None
    for r in range(rounds + 1):
        with threadpool_limits(1):
            tm = HistGradientBoostingClassifier(**team_model.HGB_PARAMS).fit(Xt, yt, sample_weight=wt)
            dm = HistGradientBoostingClassifier(**dead_model.HGB_PARAMS).fit(Xd, yd, sample_weight=wd)
        ev = team_model.evidence(tm, anti, sym)
        pd = dead_model.probability(dm, F)
        if r == rounds:
            break
        lab = np.where(pd >= 0.5, 2, np.where(ev > 0, 0, 1))
        far = _far_from_changes(lab) & graded
        idx = np.flatnonzero(far)[::SUBSAMPLE]
        sure_team = idx[(np.abs(ev[idx]) > CONF_TEAM) & (pd[idx] < 1 - CONF_DEAD)]
        sure_dead = idx[(pd[idx] > CONF_DEAD) | (pd[idx] < 1 - CONF_DEAD)]
        home = ev[sure_team] > 0
        Xt = np.vstack([Xt, np.hstack([anti[sure_team], sym[sure_team]]), np.hstack([-anti[sure_team], sym[sure_team]])])
        yt = np.concatenate([yt, home, ~home])
        wt = np.concatenate([wt, np.full(2 * len(sure_team), WEIGHT)])
        Xd = np.vstack([Xd, F[sure_dead]])
        yd = np.concatenate([yd, pd[sure_dead] > 0.5])
        wd = np.concatenate([wd, np.full(len(sure_dead), WEIGHT)])
    return ev, pd
