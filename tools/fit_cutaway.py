"""Fit how much a broadcast cutaway should move the dead-ball probability, on SkillCorner only.

    python tools/fit_cutaway.py      # after tools/bench_all.py has written out-of-fold predictions

SkillCorner's 20 matches are tracked from the TV broadcast: a frame without camera calibration
(`image_corners_projection` empty) is a replay, a close-up or graphics -- the same thing our
cutaway mask finds on BAR-ATH (mqh/broadcast.py). For each match, the dead-ball model's
out-of-fold probability (it never saw that match) is combined with the cutaway flag by a logistic
model:  logit p' = a * logit p + b * cutaway + c * cutaway_ctx + d.
Leave-one-match-out: the weights are fitted on 19 matches and scored on the 20th (dead-ball F1 and
AUC before / after), then fitted on all 20 and printed for mqh/broadcast.py (FUSION).
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mqh.external import registry  # noqa: E402

SC = os.path.join(registry.DATA, "skillcorner-opendata", "data", "matches")


def cutaway25(mid, n):
    rows = [json.loads(line) for line in open(os.path.join(SC, mid, "{}_tracking_extrapolated.jsonl".format(mid)))]
    c10 = np.array([r["image_corners_projection"]["x_top_left"] is None for r in rows])
    t = np.rint(np.arange(n) * 10 / 25).astype(int).clip(0, len(c10) - 1)
    return c10[t]


def design(p, cut):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    lg = np.log(p / (1 - p))
    ctx = uniform_filter1d(cut.astype(float), 101)      # +-2 s
    return np.column_stack([lg, cut.astype(float), ctx])


def main(profile="bar_ath"):
    data = {}
    for path in sorted(glob.glob(os.path.join(registry.CACHE, "oof-skillcorner-*-{}-s0.npz".format(profile)))):
        mid = os.path.basename(path).split("-")[2]
        z = np.load(path)
        truth = np.load(path.replace("oof-", "case-"))["truth_phases"]
        k = truth != "X"
        cut = cutaway25(mid, len(truth))
        data[mid] = (design(z["p_dead"][k], cut[k]), truth[k] == "D", z["p_dead"][k])
    print("{} SkillCorner matches with out-of-fold predictions".format(len(data)))
    before, after, auc_b, auc_a = [], [], [], []
    for mid in data:
        X = np.vstack([v[0][::5] for k, v in data.items() if k != mid])
        y = np.concatenate([v[1][::5] for k, v in data.items() if k != mid])
        m = LogisticRegression(max_iter=1000).fit(X, y)
        Xt, yt, p0 = data[mid]
        p1 = m.predict_proba(Xt)[:, 1]
        before.append(f1_score(yt, p0 >= 0.5))
        after.append(f1_score(yt, p1 >= 0.5))
        auc_b.append(roc_auc_score(yt, p0))
        auc_a.append(roc_auc_score(yt, p1))
    print("dead-ball F1 {:.3f} -> {:.3f}   AUC {:.3f} -> {:.3f}   (mean over held-out matches)".format(
        np.mean(before), np.mean(after), np.mean(auc_b), np.mean(auc_a)))
    X = np.vstack([v[0][::5] for v in data.values()])
    y = np.concatenate([v[1][::5] for v in data.values()])
    m = LogisticRegression(max_iter=1000).fit(X, y)
    print("FUSION = ({}, {:.4f})   # logit p, cutaway, cutaway +-2 s ; intercept".format(
        ", ".join("{:.4f}".format(v) for v in m.coef_[0]), m.intercept_[0]))


if __name__ == "__main__":
    main(*sys.argv[1:])
