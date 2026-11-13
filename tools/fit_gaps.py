"""Fit stream 4's shape weights (mqh/gaps.py: GapConfig.coef) on EXTERNAL matches only.

    python tools/fit_gaps.py

Metrica sample games (degraded, as in external_bench.py); target = home in possession, on live
frames of the event-derived truth. Logistic regression with NO intercept: the features are home
minus away, so the model cannot learn that one side has the ball more often. Prints the
leave-one-game-out AUC (all live frames, and inside ball gaps only), then the weights to freeze.
"""

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
from external_bench import DEFAULT_DIR, build_inputs, fetch  # noqa: E402
from mqh.external import metrica  # noqa: E402
from mqh.gaps import shape_features  # noqa: E402


def main():
    games = [metrica.load_game(n, fetch(DEFAULT_DIR)) for n in (1, 2, 3)]
    rows = {}
    for name, ball, players, team_of_slot, _, truths in build_inputs(games, [0, 1]):
        F = shape_features(players, team_of_slot, metrica.PITCH, metrica.FPS)
        t = truths["strict"]
        live = np.isin(t, ["H", "A"])
        rows[name] = (F, t, live, np.isnan(ball[:, 0]))

    def fit(items):
        X = np.vstack([F[l][::5] for F, t, l, _ in items])
        y = np.concatenate([(t[l] == "H")[::5] for F, t, l, _ in items])
        return LogisticRegression(fit_intercept=False, max_iter=2000).fit(X, y)

    print("leave-one-game-out:")
    for g in games:
        model = fit([v for k, v in rows.items() if not k.startswith(g["name"] + "-")])
        for k, (F, t, l, m) in rows.items():
            if k.startswith(g["name"] + "-"):
                s = model.decision_function(F)
                print("  {:<12} AUC live {:.3f}  in ball gaps {:.3f}".format(
                    k, roc_auc_score(t[l] == "H", s[l]), roc_auc_score(t[l & m] == "H", s[l & m])))
    model = fit(list(rows.values()))
    print("\ncoef: tuple = ({})".format(", ".join("{:.4f}".format(v) for v in model.coef_[0])))


if __name__ == "__main__":
    main()
