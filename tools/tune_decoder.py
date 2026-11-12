"""Choose decoder constants on EXTERNAL cases only, leave-one-match-out, with a noise rule.

    python tools/tune_decoder.py --cases .cache/external_cases        # real use: external matches
    python tools/tune_decoder.py --synthetic                          # mechanism demo, made-up matches

A case is an .npz with three arrays of video length: `evidence` (float, + home / - away),
`dead` (bool) and `truth` (str 'H'/'A'/'D'/'X', X = not graded). Build them from external matches
(Metrica / IDSSE) degraded to look like our tracking; never from BAR-ATH's ground truth or video
(RULES.md). Scores come from the official scorer code.

Rules, from the team plan and docs/DECISIONS.md:
  - the metric is spellF1, the one that ranks;
  - a setting only replaces the default if it beats it by more than the noise between cases
    (standard error over cases, floored at 0.005);
  - among settings within one noise unit of the best, take the one nearest the default;
  - report the leave-one-match-out score, i.e. each case scored with constants chosen on the others.
"""

import argparse
import glob
import itertools
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
import evaluation.possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from mqh.decoder import DecoderConfig, decode_half  # noqa: E402

GRID = {
    "switch_penalty": [10.0, 20.0, 30.0, 45.0, 60.0],
    "restart_switch_scale": [0.3, 0.5, 0.7, 1.0],
    "restart_min_dead_frames": [15, 25, 50],
}
NOISE_FLOOR = 0.005
ABLATION = [
    ("plain Viterbi", dict(place_switch_in_gaps=False, restart_switch_scale=1.0, clean_dead_mask=False)),
    ("+ clean dead mask", dict(place_switch_in_gaps=False, restart_switch_scale=1.0)),
    ("+ place in gaps", dict(restart_switch_scale=1.0)),
    ("+ cheaper restart (default)", dict()),
]


def decode_case(case, cfg):
    labels = np.full(len(case["evidence"]), "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        labels[first:last + 1] = decode_half(case["evidence"], case["dead"], first, last, cfg)
    return labels


def spell_f1(truth, labels):
    gt = pm.label_codes(truth)
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pm.label_codes(labels), runs), runs)["mean_f1"]


def score(case, cfg):
    return spell_f1(case["truth"], decode_case(case, cfg))


def load_cases(folder):
    cases = []
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        z = np.load(path, allow_pickle=False)
        cases.append({"name": os.path.basename(path), "evidence": z["evidence"].astype(float),
                      "dead": z["dead"].astype(bool), "truth": z["truth"].astype("<U1")})
    return cases


def synthetic_cases(seeds):
    import synth
    from mqh.pipeline import Config, prepare
    cases = []
    for seed in seeds:
        with tempfile.TemporaryDirectory() as tmp:
            truth = synth.write(tmp, turnover_frac=0.4, seed=seed, dead_flip_prob=0.7)
            evidence, dead, _, _ = prepare(os.path.join(tmp, "synth_atd.txt"),
                                           os.path.join(tmp, "synth_atd.xml"), Config(), hooks={})
        cases.append({"name": "synthetic-{}".format(seed), "evidence": evidence, "dead": dead,
                      "truth": truth})
    return cases


def configs():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, values))


def distance_from_default(params):
    default = DecoderConfig()
    return sum(getattr(default, k) != v for k, v in params.items())


def pick(table, idx):
    """Index of the chosen config using only the cases in `idx`: nearest the default among the
    settings within one noise unit of the best."""
    mean = table[:, idx].mean(axis=1)
    se = table[:, idx].std(axis=1, ddof=1) / np.sqrt(len(idx)) if len(idx) > 1 else np.zeros(len(mean))
    noise = max(NOISE_FLOOR, float(se[np.argmax(mean)]))
    ok = np.flatnonzero(mean >= mean.max() - noise)
    best = sorted(ok, key=lambda i: (DISTANCE[i], -mean[i]))[0]
    return int(best), float(mean.max()), noise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", help="folder of external .npz cases")
    parser.add_argument("--synthetic", action="store_true", help="made-up matches (mechanism demo)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    args = parser.parse_args()
    if bool(args.cases) == bool(args.synthetic):
        parser.error("give exactly one of --cases or --synthetic")
    cases = synthetic_cases(args.seeds) if args.synthetic else load_cases(args.cases)
    if len(cases) < 2:
        parser.error("need at least two cases for leave-one-match-out")

    grid = list(configs())
    global DISTANCE
    DISTANCE = [distance_from_default(g) for g in grid]
    table = np.array([[score(c, DecoderConfig(**g)) for c in cases] for g in grid])
    default_i = [i for i, g in enumerate(grid) if DISTANCE[i] == 0][0]

    if args.synthetic:
        print("SYNTHETIC DEMO: made-up evidence is cleaner than the real tracking, so these numbers"
              " only show the procedure works. Do not adopt them.\n")
    print("cases: " + ", ".join(c["name"] for c in cases))
    print("\nAblation at the default constants (spellF1 per case):")
    for name, kw in ABLATION:
        s = [score(c, DecoderConfig(**kw)) for c in cases]
        print("  {:<30}{:>8.3f}   ".format(name, np.mean(s)) + " ".join("{:.3f}".format(x) for x in s))

    held, dflt, oracle = [], [], []
    for h in range(len(cases)):
        rest = [i for i in range(len(cases)) if i != h]
        chosen, _, _ = pick(table, rest)
        held.append(table[chosen, h])
        dflt.append(table[default_i, h])
        oracle.append(table[:, h].max())
    print("\nLeave-one-match-out spellF1: chosen on the others {:.3f} | default {:.3f} | "
          "oracle per case {:.3f}".format(np.mean(held), np.mean(dflt), np.mean(oracle)))

    chosen, best, noise = pick(table, list(range(len(cases))))
    gain = table[chosen].mean() - table[default_i].mean()
    print("\nAll cases: best mean {:.3f}, noise {:.3f}".format(best, noise))
    print("nearest-default setting inside the noise band:", grid[chosen])
    if grid[chosen] == grid[default_i] or gain <= noise:
        print("Recommendation: keep the defaults (gain {:+.3f} does not exceed the noise {:.3f})."
              .format(gain, noise))
    else:
        print("Recommendation: consider {} (gain {:+.3f} exceeds the noise {:.3f}); record it in "
              "docs/DECISIONS.md with the date and these numbers.".format(grid[chosen], gain, noise))


if __name__ == "__main__":
    main()
