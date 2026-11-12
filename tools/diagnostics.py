"""Sanity numbers for a submission file, per half. Reads only the label CSV, never match data.

    python tools/diagnostics.py submissions/<team_name>.csv

Prints, for each graded half: label shares, number of possessions (maximal runs of H or of A; dead
ball breaks a run), their duration statistics, and how many are suspiciously short. Compare the
shape across halves and against what you know of football (organisers' docs: about 14 s mean in
the public window, which is a sanity range and not a target). Do not tune against the public
ground truth.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evaluation.possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402


def runs_of(labels):
    """[(label, length)] run-length encoding."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return [(labels[s], int(e - s)) for s, e in zip(starts, ends)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--short-s", type=float, default=3.0, help="count possessions below this")
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    table = pm.read_label_csv(args.csv)
    labels = np.array([str(v).strip().upper() for v in table["label"].values])
    print("{}  ({} rows)".format(os.path.basename(args.csv), len(labels)))

    for half, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        n = len(seg)
        shares = {lab: float(np.mean(seg == lab)) for lab in "HADX"}
        poss = [length / fps for lab, length in runs_of(seg) if lab in ("H", "A")]
        print("\nhalf {}  frames {}..{}".format(half, first, last))
        print("  shares   " + "  ".join("{} {:.1%}".format(k, v) for k, v in shares.items()))
        if not poss:
            print("  no possessions")
            continue
        poss = np.array(poss)
        print("  possessions {}   mean {:.1f} s   median {:.1f} s   min {:.1f} s   max {:.1f} s".format(
            len(poss), poss.mean(), np.median(poss), poss.min(), poss.max()))
        print("  shorter than {:.0f} s: {} ({:.0%})   longer than 60 s: {}".format(
            args.short_s, int((poss < args.short_s).sum()),
            float((poss < args.short_s).mean()), int((poss > 60).sum())))
        dead_runs = [length / fps for lab, length in runs_of(seg) if lab == "D"]
        if dead_runs:
            print("  dead-ball stretches {}   mean {:.1f} s   longest {:.1f} s".format(
                len(dead_runs), np.mean(dead_runs), np.max(dead_runs)))
        if shares["X"] > 0:
            print("  note: X on graded frames counts as a miss for macroF1")

    outside = np.ones(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        outside[first:last + 1] = False
    other = set(np.unique(labels[outside])) - {"X"}
    if other:
        print("\nnote: labels other than X outside the graded halves ({}) are ignored by the "
              "scorer".format(sorted(other)))


if __name__ == "__main__":
    main()
