"""Does self-training help? Checked on external matches before it is allowed near BAR-ATH.

    python tools/check_self_training.py          # after tools/build_external_cache.py

Self-training: train the team and dead-ball models, predict a new match, keep the frames the models
are very sure about (pseudo-labels produced by the algorithm, never by a person: allowed by
RULES.md), add them to the training set, retrain, predict again. The idea is to adapt the models to
the target's own tracking. It can also reinforce the models' own mistakes, so it is tested here:
each target is an external match held out of training (two per provider), scored with the official
scorer before and after, with the same decoder.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
from bench_all import FOLDS, cases, fold_of, score, training_rows  # noqa: E402
from mqh import dead_model, team_model, self_train  # noqa: E402
from mqh.external import registry  # noqa: E402
from mqh.pipeline import Config, decode  # noqa: E402


def main():
    test = cases("bar_ath")
    allc = [c for prof in ("bar_ath", "match_like") for c in cases(prof)]
    picks = []
    for p in registry.PROVIDERS:
        picks += [c for c in test if c[0] == p][:2]
    cfg = Config()
    before, after = [], []
    for provider, match, path in picks:
        f = fold_of(provider, match)
        train = [c[2] for c in allc if fold_of(c[0], c[1]) != f]
        team_rows, dead_rows = training_rows(train)
        z = np.load(path)
        target = dict(anti=z["anti"].astype(float), sym=z["sym"].astype(float), deadF=z["deadF"])
        blocks = [(i + 1, int(a), int(b)) for i, (a, b) in enumerate(z["blocks"])]
        n = len(z["evidence"])
        truth = z["truth_" + registry.PROVIDERS[provider]["conventions"][0]]
        graded = truth != "X"
        res = []
        for rounds in (0, 1):
            ev, pd = self_train.fit_predict(team_rows, dead_rows, target, graded, rounds=rounds)
            lab = decode(ev, pd >= 0.5, blocks, n, cfg.decoder, (z["touch_frames"], z["touch_teams"]))
            res.append(score(truth, lab)[0])
        before.append(res[0])
        after.append(res[1])
        print("{:<12} {:<8} spellF1 {:.3f} -> {:.3f}".format(provider, match, res[0], res[1]), flush=True)
    d = np.array(after) - np.array(before)
    print("mean {:.3f} -> {:.3f}  (change {:+.3f} ± {:.3f})".format(np.mean(before), np.mean(after), d.mean(),
                                                                d.std(ddof=1) / np.sqrt(len(d))))


if __name__ == "__main__":
    main()
