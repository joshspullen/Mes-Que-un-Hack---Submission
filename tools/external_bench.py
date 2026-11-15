"""Score the whole pipeline on EXTERNAL matches, degraded to look like our tracking.

    python tools/external_bench.py                     # downloads Metrica's sample data once
    python tools/external_bench.py --write-cases DIR   # also writes cases for tools/tune_decoder.py

Metrica Sports sample data (https://github.com/metrica-sports/sample-data, 3 anonymised games;
please acknowledge the source). Labels come from their event data by code (mqh/external/metrica.py),
under two readings of "control" (loose, strict). Nothing here reads BAR-ATH's ground truth or video.

For each game x degradation seed the same pipeline code as make_submission.py runs
(mqh.pipeline.prepare_arrays + decode) and is scored with the official scorer. Prints spellF1,
macroF1, dead-ball F1 and possession counts, per case and on average. Use it to compare variants:
a change is only adopted if its mean spellF1 gain exceeds the spread between cases.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evaluation.possession_metrics as pm  # noqa: E402
from mqh.external import metrica  # noqa: E402
from mqh.external.degrade import PROFILES, DegradeConfig, degrade  # noqa: E402
from mqh.pipeline import Config, decode, prepare_arrays  # noqa: E402

DEFAULT_DIR = os.path.join(ROOT, "data", "external", "metrica-sample-data")
REPO = "https://github.com/metrica-sports/sample-data"


def fetch(folder):
    if not os.path.isdir(os.path.join(folder, "data")):
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, folder])
    return folder


def build_inputs(games, seeds, dcfg=DegradeConfig()):
    """[(name, ball, players, team_of_slot, blocks, truth_by_convention)] degraded copies."""
    out = []
    for g in games:
        truths = {c: metrica.labels(g, c) for c in ("loose", "strict")}
        live = np.isin(truths["loose"], ["H", "A"])
        blocks = metrica.blocks_of(g)
        for seed in seeds:
            ball, players = degrade(g["ball"], g["players"], live, seed=seed, cfg=dcfg)
            out.append(("{}-s{}".format(g["name"], seed), ball, players, g["team_of_slot"], blocks, truths))
    return out


def scores(truth, labels):
    gt = pm.label_codes(truth)
    pred = pm.label_codes(labels)
    r = pm.evaluate(gt, pred)
    runs = pm.valid_runs(gt)
    stats = pm.possession_stats(pred, runs)
    cm = pm.confusion(gt, pred)
    d = pm.per_class_prf(pm.scored_confusion(cm), pm.D)[2]
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "dF1": d,
            "poss": stats["n_spells"], "dur": stats["mean_duration_s"],
            "true_poss": pm.possession_stats(gt, runs)["n_spells"]}


def run_case(case, cfg, hooks, team_model=None):
    name, ball, players, team_of_slot, blocks, truths = case
    evidence, dead, blocks, n, touches = prepare_arrays(
        ball, players, team_of_slot, cfg, metrica.PITCH, metrica.FPS, blocks, hooks=hooks,
        team_model=team_model)
    labels = decode(evidence, dead, blocks, n, cfg.decoder, touches)
    return {c: scores(t, labels) for c, t in truths.items()}, (evidence, dead, blocks)


def logo_models(games, cfg):
    """{game name: team model trained on the OTHER games} (stream 3c, leave-one-game-out)."""
    from mqh import team_model as tm
    from mqh.external.degrade import PROFILES

    rows = {g["name"]: tm.training_rows([g], [PROFILES[p] for p in tm.TRAIN_PROFILES], [0, 1],
                                        cfg, metrica.PITCH, metrica.FPS) for g in games}
    return {name: tm.fit([r for other, rs in rows.items() if other != name for r in rs]) for name in rows}


def bench(cases, cfg=Config(), hooks=None, verbose=True, label="", models=None):
    rows = []
    for case in cases:
        model = models[case[0].split("-")[0]] if models else None
        res, _ = run_case(case, cfg, hooks, model)
        rows.append((case[0], res))
        if verbose:
            r = res["strict"]
            print("  {:<14} spell {:.3f} (loose {:.3f})  macro {:.3f}  dF1 {:.3f}  poss {:>4} / {:>4}  dur {:5.1f}s".format(
                case[0], r["spell"], res["loose"]["spell"], r["macro"], r["dF1"], r["poss"], r["true_poss"], r["dur"]))
    summary = {}
    for conv in ("strict", "loose"):
        for k in ("spell", "macro", "dF1"):
            v = np.array([res[conv][k] for _, res in rows])
            summary[(conv, k)] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    print("{:<34} spellF1 {:.3f} ±{:.3f} (loose {:.3f})  macroF1 {:.3f}  dead F1 {:.3f}".format(
        label or "mean", *summary[("strict", "spell")], summary[("loose", "spell")][0],
        summary[("strict", "macro")][0], summary[("strict", "dF1")][0]))
    return summary


def write_cases(cases, cfg, hooks, folder, models=None):
    os.makedirs(folder, exist_ok=True)
    for case in cases:
        model = models[case[0].split("-")[0]] if models else None
        _, (evidence, dead, blocks) = run_case(case, cfg, hooks, model)
        for conv, truth in case[5].items():
            np.savez_compressed(os.path.join(folder, "{}-{}.npz".format(case[0], conv)),
                                evidence=evidence, dead=dead, truth=truth,
                                blocks=np.array([[b[1], b[2]] for b in blocks]))
    print("wrote", len(cases) * 2, "cases to", folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--write-cases", metavar="DIR")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="default",
                        help="degradation profile (mqh/external/degrade.py)")
    parser.add_argument("--learned", action="store_true",
                        help="add stream 3c, each game scored by a model trained on the other two")
    args = parser.parse_args()

    root = fetch(args.dir)
    games = [metrica.load_game(n, root) for n in args.games]
    cases = build_inputs(games, args.seeds, PROFILES[args.profile])
    print("external bench: {} cases (Metrica sample games {}, seeds {}, profile {})".format(
        len(cases), args.games, args.seeds, args.profile))
    models = logo_models(games, Config()) if args.learned else None
    bench(cases, Config(), hooks=None, label="current pipeline" + (" + learned team" if models else ""),
          models=models)
    if args.write_cases:
        write_cases(cases, Config(), None, args.write_cases, models)


if __name__ == "__main__":
    main()
