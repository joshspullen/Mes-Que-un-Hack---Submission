"""Stream 3c: learned team evidence -- which team controls the ball, from many weak signals at once.

The hand-built score of stream 3 (mqh/decode.team_evidence) gets the team right on about 67% of
live frames of the degraded external games; with the ball 2 m off and lost 41% of the time, one
rule cannot weigh distances, players nearby, touches, team shape and context together. A gradient
boosted model does, and gets 85-88% (leave-one-game-out). On the external bench it moves spellF1
from 0.360 to 0.426 under the match-like profile, better on all 6 cases (docs/DECISIONS.md).

Training data: EXTERNAL matches only, Metrica's three public sample games, degraded with both
profiles of mqh/external/degrade.py; labels from their events by code (mqh/external/metrica.py,
`strict` reading). The hackathon match's ground truth and video are never read. The model is
trained at run time from the pinned sample-data commit, single-threaded and seeded, so the same
environment gives the same file (tools/repro_check.py checks it).

Features are split in two groups so the model cannot favour a side: ANTISYMMETRIC ones (home minus
away: change sign when the teams are swapped) and SYMMETRIC ones (the same for both teams). Every
training frame is also shown with the teams swapped, and the prediction averages the model on the
frame and on its swap, so evidence(swap) = -evidence exactly.
"""

import os
import pickle

import numpy as np
from scipy.ndimage import uniform_filter1d

from mqh.gaps import shape_features
from mqh.touches import touch_events

SAMPLE_DATA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"   # metrica-sports/sample-data
SUBSAMPLE = 5                  # every 5th frame for training: neighbours are near-duplicates
HGB_PARAMS = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                  early_stopping=False, random_state=0)


def team_features(ball, players, team_of_slot, pitch, fps, base_evidence):
    """(antisymmetric (n, 15), symmetric (n, 5)) feature arrays, finite everywhere."""
    scale = np.asarray(pitch, dtype=float)
    team = np.asarray(team_of_slot)
    n = len(ball)
    xy = ball * scale
    p = players * scale
    d = np.hypot(*np.moveaxis(p - xy[:, None, :], -1, 0))
    d = np.where(np.isnan(d), 99.0, d)
    dh = np.sort(d[:, team == 0], axis=1)
    da = np.sort(d[:, team == 1], axis=1)
    seen = ~np.isnan(ball[:, 0])

    def clip(x):
        return np.clip(np.where(seen, x, 30.0), 0.0, 30.0)

    h1, h2, a1, a2 = clip(dh[:, 0]), clip(dh[:, 1]), clip(da[:, 0]), clip(da[:, 1])
    nh2 = np.where(seen, (dh < 2).sum(axis=1), 0)
    na2 = np.where(seen, (da < 2).sum(axis=1), 0)
    nh5 = np.where(seen, (dh < 5).sum(axis=1), 0)
    na5 = np.where(seen, (da < 5).sum(axis=1), 0)
    k = max(1, int(round(0.2 * fps)))
    speed = np.zeros(n)
    speed[k:-k] = np.nan_to_num(np.hypot(*(xy[2 * k:] - xy[:-2 * k]).T) * fps / (2 * k))
    shape = shape_features(players, team, pitch, fps)

    frames, teams = touch_events(ball, players, team, pitch, fps)
    last = np.zeros(n)
    since = np.full(n, 10.0)
    nxt = np.zeros(n)
    until = np.full(n, 10.0)
    idx = np.arange(n)
    if len(frames):
        before = np.searchsorted(frames, idx, side="right") - 1
        ok = before >= 0
        last[ok] = np.where(teams[before[ok]] == 0, 1.0, -1.0)
        since[ok] = np.minimum((idx[ok] - frames[before[ok]]) / fps, 10.0)
        after = np.searchsorted(frames, idx, side="left")
        ok = after < len(frames)
        nxt[ok] = np.where(teams[after[ok]] == 0, 1.0, -1.0)
        until[ok] = np.minimum((frames[after[ok]] - idx[ok]) / fps, 10.0)

    margin = a1 - h1
    s1, s3, s10 = int(fps) | 1, int(3 * fps) | 1, int(10 * fps) | 1
    anti = np.column_stack([
        h1 - a1, h2 - a2, nh2 - na2, nh5 - na5, base_evidence,
        shape[:, 0], shape[:, 1], shape[:, 2], last, nxt,
        uniform_filter1d(margin, s1), uniform_filter1d(margin, s3), uniform_filter1d(margin, s10),
        uniform_filter1d(base_evidence, s3), uniform_filter1d(base_evidence, s10),
    ])
    sym = np.column_stack([speed, seen.astype(float), np.minimum(h1, a1), np.log1p(since), np.log1p(until)])
    return anti, sym


def _xy(anti, sym):
    return np.hstack([anti, sym])


def fit(rows):
    """rows: [(anti, sym, truth)] -> fitted model. Live frames only, each also with teams swapped."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    X, y = [], []
    for anti, sym, truth in rows:
        live = np.flatnonzero(np.isin(truth, ["H", "A"]))[::SUBSAMPLE]
        home = truth[live] == "H"
        X += [_xy(anti[live], sym[live]), _xy(-anti[live], sym[live])]
        y += [home, ~home]
    with threadpool_limits(1):
        return HistGradientBoostingClassifier(**HGB_PARAMS).fit(np.vstack(X), np.concatenate(y))


def evidence(model, anti, sym):
    """Antisymmetric evidence in [-1, 1]: 2 P(home) - 1, averaged with the teams-swapped frame."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        p = model.predict_proba(_xy(anti, sym))[:, 1]
        q = model.predict_proba(_xy(-anti, sym))[:, 1]
    return np.clip(p - q, -1.0, 1.0)      # mean of the two votes: ((2p - 1) + (1 - 2q)) / 2


def training_rows(games, profiles, seeds, cfg, pitch, fps):
    """Feature rows for external games, built through the same pipeline code as the submission."""
    from mqh.external import metrica
    from mqh.external.degrade import degrade
    from mqh.pipeline import discover_hooks, prepare_arrays

    hooks = discover_hooks()   # Smart Tagging hooks skip themselves: external games have no tags
    rows = []
    for g in games:
        truth = metrica.labels(g, "strict")
        live = np.isin(truth, ["H", "A"])
        for profile in profiles:
            for seed in seeds:
                ball, players = degrade(g["ball"], g["players"], live, seed=seed, cfg=profile)
                base, _, _, _, _ = prepare_arrays(ball, players, g["team_of_slot"], cfg, pitch, fps,
                                                  metrica.blocks_of(g), hooks=hooks)
                anti, sym = team_features(ball, players, g["team_of_slot"], pitch, fps, base)
                rows.append((anti, sym, truth))
    return rows


def default_model(cfg, cache_dir):
    """Train on all three Metrica sample games, both degradation profiles, seeds 0 and 1; cached."""
    from mqh.external import metrica
    from mqh.external.degrade import PROFILES

    path = os.path.join(cache_dir, "team_model-{}.pkl".format(_code_key()))
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    root = fetch_pinned()
    games = [metrica.load_game(k, root) for k in (1, 2, 3)]
    rows = training_rows(games, [PROFILES["default"], PROFILES["match_like"]], [0, 1], cfg,
                         metrica.PITCH, metrica.FPS)
    model = fit(rows)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)
    return model


def _code_key():
    """Cache key: the code that shapes the model and the library that fits it."""
    import hashlib

    import sklearn

    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256(sklearn.__version__.encode())
    for root, _, files in sorted(os.walk(here)):
        for name in sorted(files):
            if name.endswith(".py"):
                with open(os.path.join(root, name), "rb") as handle:
                    h.update(handle.read())
    return h.hexdigest()[:12]


def fetch_pinned():
    """Metrica's sample data at the pinned commit, downloaded once into data/external/ (git-ignored)."""
    import subprocess

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "external", "metrica-sample-data")
    if not os.path.isdir(os.path.join(root, "data")):
        os.makedirs(os.path.dirname(root), exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1",
                               "https://github.com/metrica-sports/sample-data", root])
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"]).decode().strip()
    if head != SAMPLE_DATA_COMMIT:
        subprocess.check_call(["git", "-C", root, "fetch", "--depth", "1", "origin", SAMPLE_DATA_COMMIT])
        subprocess.check_call(["git", "-C", root, "checkout", "-q", SAMPLE_DATA_COMMIT])
    return root
