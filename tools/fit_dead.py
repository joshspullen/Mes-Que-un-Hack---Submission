"""Fit stream 2's dead-ball weights (mqh/dead.py: DeadConfig) on EXTERNAL matches only.

    python tools/fit_dead.py                 # leave-one-game-out scores, then the weights to freeze

Metrica Sports sample data (3 games; please acknowledge the source), degraded with
mqh/external/degrade.py so the features look like our tracking's. Labels: dead = 'D' in the
event-derived truth (mqh/external/metrica.py). The hackathon match is never read.

Prints, per held-out game, the frame AUC and the dead-ball F1 at the frozen threshold, then the
weights fitted on all three games, ready to paste into DeadConfig. Deterministic.
"""

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
from external_bench import DEFAULT_DIR, build_inputs, fetch  # noqa: E402
from mqh.dead import FEATURES, DeadConfig, dead_features  # noqa: E402
from mqh.external import metrica  # noqa: E402

SAMPLE = 5   # every 5th frame: neighbouring frames are near-duplicates

# Left out of the frozen model, with the reason:
#   no_ball, gap_log -- Metrica's sample tracking hides the ball on 90-99% of dead frames, so these
#       learn "no ball = stoppage", a habit of that provider we cannot assume for BAR-ATH's
#       automatic tracking (whose ball also vanishes in crowds and in the air). Without them the
#       leave-one-game-out AUC barely moves (0.93/0.93/0.92 -> 0.92/0.92/0.92).
#   walk_ratio -- nearly collinear with walk_long; with both the weights take opposite signs
#       (+8.4 / -7.4), which would not transfer. Dropping it costs nothing measurable.
DROPPED = ("walk_ratio", "no_ball", "gap_log")


def table(cases, cfg):
    rows = {}
    for name, ball, players, _, _, truths in cases:
        X = dead_features(ball, players, metrica.PITCH, metrica.FPS, cfg)
        truth = truths["strict"]
        k = np.flatnonzero(truth != "X")
        rows[name] = (X[k], (truth[k] == "D").astype(int))
    return rows


def fit(X, y):
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(X[::SAMPLE], y[::SAMPLE])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--drop", nargs="*", default=list(DROPPED), help="features to leave out")
    args = parser.parse_args()

    cfg = DeadConfig()
    games = [metrica.load_game(n, fetch(args.dir)) for n in (1, 2, 3)]
    rows = table(build_inputs(games, args.seeds), cfg)
    keep = [i for i, f in enumerate(FEATURES) if f not in args.drop]

    print("leave-one-game-out (train on two games, test on the third):")
    for g in games:
        train = [v for k, v in rows.items() if not k.startswith(g["name"] + "-")]
        test = [v for k, v in rows.items() if k.startswith(g["name"] + "-")]
        model = fit(np.vstack([x for x, _ in train])[:, keep], np.concatenate([y for _, y in train]))
        X, y = np.vstack([x for x, _ in test])[:, keep], np.concatenate([y for _, y in test])
        p = model.predict_proba(X)[:, 1]
        print("  {:<10} AUC {:.3f}  dead F1 @0.5 {:.3f}  dead share true {:.2f} / predicted {:.2f}".format(
            g["name"], roc_auc_score(y, p), f1_score(y, p >= 0.5), y.mean(), (p >= 0.5).mean()))

    model = fit(np.vstack([x for x, _ in rows.values()])[:, keep],
                np.concatenate([y for _, y in rows.values()]))
    w = np.zeros(len(FEATURES))
    w[keep] = model.coef_[0]
    print("\nweights fitted on all games (paste into DeadConfig):")
    print("    weights: tuple = ({})".format(", ".join("{:.4f}".format(v) for v in w)))
    print("    intercept: float = {:.4f}".format(model.intercept_[0]))
    for f, v in zip(FEATURES, w):
        print("      {:<12} {:+.3f}".format(f, v))


if __name__ == "__main__":
    main()
