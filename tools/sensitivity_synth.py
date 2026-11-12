"""How much does each decoder idea matter, and how flat is the switch penalty? Synthetic only.

    python tools/sensitivity_synth.py [--seeds 7 11 23] [--turnover 0.4]

Builds made-up matches with known possessions (tests/synth.py), runs the evidence step once per
match, then decodes with different settings and scores spellF1 with the official scorer code.
This is a robustness check of the *mechanism*; it says nothing about the real match and the
numbers must not be used to pick constants for it. Real constants are reasoned (docs/DECISIONS.md).
"""

import argparse
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
import synth  # noqa: E402
import evaluation.possession_metrics as pm  # noqa: E402
from mqh.decoder import DecoderConfig  # noqa: E402
from mqh.pipeline import Config, decode, prepare  # noqa: E402

PENALTIES = [10, 20, 30, 45, 60, 90]
VARIANTS = [
    ("plain Viterbi", dict(place_switch_in_gaps=False, restart_switch_scale=1.0)),
    ("+ place in gaps", dict(place_switch_in_gaps=True, restart_switch_scale=1.0)),
    ("+ cheaper restart (default)", dict(place_switch_in_gaps=True, restart_switch_scale=0.5)),
]


def spell_f1(truth, labels):
    gt = pm.label_codes(truth)
    pred = pm.label_codes(labels)
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--turnover", type=float, default=0.4,
                        help="fraction of possession changes with no dead ball in between")
    parser.add_argument("--dead-flip", type=float, default=1.0,
                        help="probability that possession changes team across a stoppage")
    args = parser.parse_args()

    table = {(v[0], p): [] for v in VARIANTS for p in PENALTIES}
    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as tmp:
            truth = synth.write(tmp, turnover_frac=args.turnover, seed=seed,
                                dead_flip_prob=args.dead_flip)
            evidence, dead, blocks, n = prepare(os.path.join(tmp, "synth_atd.txt"),
                                                os.path.join(tmp, "synth_atd.xml"), Config(), hooks={})
        for name, kw in VARIANTS:
            for pen in PENALTIES:
                labels = decode(evidence, dead, blocks, n, DecoderConfig(switch_penalty=pen, **kw))
                table[(name, pen)].append(spell_f1(truth, labels))

    print("spellF1 on synthetic matches (mean over seeds {}; turnover {:.0%}; "
          "team changes across a stoppage {:.0%})".format(args.seeds, args.turnover, args.dead_flip))
    print("{:<30}".format("penalty") + "".join("{:>8}".format(p) for p in PENALTIES))
    for name, _ in VARIANTS:
        print("{:<30}".format(name) + "".join(
            "{:>8.3f}".format(np.mean(table[(name, p)])) for p in PENALTIES))


if __name__ == "__main__":
    main()
