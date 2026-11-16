"""The full bench: 30 external matches, 3 providers, grouped cross-validation, official scorer.

    python tools/bench_all.py                    # needs tools/build_external_cache.py first
    python tools/bench_all.py --profile match_like

Matches are split into 5 folds (every match's frames stay in one fold, providers spread over the
folds). For each fold, the team model and the dead-ball model are trained on the other four folds
(both degradation profiles), then the held-out matches are decoded and scored with
evaluation/possession_metrics.py against every label convention of their provider. Reported per
provider and overall, mean and standard error over matches. Variants compared side by side.
"""

import argparse
import glob
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
import evaluation.possession_metrics as pm  # noqa: E402
from mqh import dead_model, team_model  # noqa: E402
from mqh.external import registry  # noqa: E402
from mqh.pipeline import Config, decode  # noqa: E402

FOLDS = 5


def cases(profile):
    out = []
    for path in sorted(glob.glob(os.path.join(registry.CACHE, "case-*-{}-s*.npz".format(profile)))):
        _, provider, match = os.path.basename(path).split("-")[:3]
        out.append((provider, match, path))
    return out


def fold_of(provider, match):
    """Deterministic: matches of each provider dealt round-robin over the folds."""
    same = sorted(str(m) for p, m in registry.all_matches() if p == provider)
    offset = list(registry.PROVIDERS).index(provider)
    return (same.index(str(match)) + offset) % FOLDS


def training_rows(paths):
    team_rows, dead_rows = [], []
    for path in paths:
        z = np.load(path)
        truth = [z[k] for k in z.files if k.startswith("truth_")][0]
        k = np.flatnonzero(truth != "X")[::5]
        t = truth[k]
        team_rows.append((z["anti"][k], z["sym"][k], t))
        dead_rows.append((z["deadF"][k], t))
    return team_rows, dead_rows


def fit_fold(train_paths):
    team_rows, dead_rows = training_rows(train_paths)
    team_model.SUBSAMPLE, dead_model.SUBSAMPLE = 1, 1     # rows are already every 5th frame
    try:
        return team_model.fit(team_rows), dead_model.fit(dead_rows)
    finally:
        team_model.SUBSAMPLE, dead_model.SUBSAMPLE = 5, 5


def score(truth, labels):
    gt, pred = pm.label_codes(truth), pm.label_codes(labels)
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"]


def variants(z, tm, dm, cfg):
    """{name: labels} for one held-out case."""
    blocks = [(i + 1, int(a), int(b)) for i, (a, b) in enumerate(z["blocks"])]
    n = len(z["evidence"])
    touches = (z["touch_frames"], z["touch_teams"])
    learned_ev = team_model.evidence(tm, z["anti"].astype(float), z["sym"].astype(float))
    p_dead = dead_model.probability(dm, z["deadF"])
    out = {}
    out["hand team + dead v2"] = decode(z["evidence"].astype(float), z["dead_v2"], blocks, n, cfg.decoder, touches)
    out["learned team + dead v2"] = decode(learned_ev, z["dead_v2"], blocks, n, cfg.decoder, touches)
    for th in (0.4, 0.5, 0.6):
        out["learned team + learned dead @{}".format(th)] = decode(learned_ev, p_dead >= th, blocks, n, cfg.decoder, touches)
    return out, learned_ev, p_dead


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="bar_ath")
    parser.add_argument("--train-profiles", nargs="+", default=["bar_ath", "match_like"])
    args = parser.parse_args()

    test = cases(args.profile)
    allc = [c for prof in args.train_profiles for c in cases(prof)]
    cfg = Config()
    results = {}     # variant -> [(provider, match, conv, spell, macro)]
    for f in range(FOLDS):
        held = [c for c in test if fold_of(c[0], c[1]) == f]
        train = [c[2] for c in allc if fold_of(c[0], c[1]) != f]
        tm, dm = fit_fold(train)
        for provider, match, path in held:
            z = np.load(path)
            labs, learned_ev, p_dead = variants(z, tm, dm, cfg)
            np.savez_compressed(path.replace("case-", "oof-"), p_home=((learned_ev + 1) / 2).astype(np.float32),
                                p_dead=p_dead.astype(np.float32))
            for conv in [k for k in z.files if k.startswith("truth_")]:
                for name, lab in labs.items():
                    s, m = score(z[conv], lab)
                    results.setdefault(name, []).append((provider, match, conv[6:], s, m))
        print("fold {} done ({} held-out matches)".format(f, len(held)), flush=True)

    print("\nprofile {}  (main convention per provider; spellF1 mean ± s.e. over matches)".format(args.profile))
    provs = list(registry.PROVIDERS)
    head = "{:<40}".format("variant") + "".join("{:>18}".format(p) for p in provs) + "{:>18}{:>10}".format("all (spell)", "macro")
    print(head)
    for name, rows in results.items():
        main = [r for r in rows if r[2] == registry.PROVIDERS[r[0]]["conventions"][0]]
        cells = []
        for p in provs:
            v = np.array([r[3] for r in main if r[0] == p])
            cells.append("{:.3f} ±{:.3f}".format(v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else "-")
        # overall: each provider weighs the same
        overall = np.mean([np.mean([r[3] for r in main if r[0] == p]) for p in provs if any(r[0] == p for r in main)])
        macro = np.mean([np.mean([r[4] for r in main if r[0] == p]) for p in provs if any(r[0] == p for r in main)])
        print("{:<40}".format(name) + "".join("{:>18}".format(c) for c in cells) + "{:>18.3f}{:>10.3f}".format(overall, macro))


if __name__ == "__main__":
    main()
