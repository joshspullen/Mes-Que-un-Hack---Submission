"""Run the pipeline on the synthetic match and score it with the official scorer.

    python tests/test_pipeline.py

Checks the plumbing (row count, alphabet, graded frames only) and that the decoder clearly beats
the nearest-player baseline on a world where the answer is known. It says nothing about how well
the method does on the real match.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import synth  # noqa: E402
from mqh.pipeline import Config, run, write_csv  # noqa: E402
import evaluation.possession_metrics as pm  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        truth = synth.write(tmp)
        raw, meta = os.path.join(tmp, "synth_atd.txt"), os.path.join(tmp, "synth_atd.xml")
        labels = run(raw, meta, Config())

        assert labels.shape == (synth.N,), labels.shape
        assert set(np.unique(labels)) <= set("HADX")
        graded = truth != "X"
        assert (labels[~graded] == "X").all(), "wrote labels outside the graded halves"
        assert (labels[graded] != "X").all(), "left X on graded frames"

        results = os.path.join(tmp, "results")
        os.makedirs(results)
        write_csv(os.path.join(results, "decoder_v0.csv"), labels)
        # the plain nearest-player vote with a +-50 frame window, for comparison
        write_csv(os.path.join(results, "decoder_noswitch.csv"),
                  run(raw, meta, Config(switch_penalty=0.0)))

        gt = os.path.join(tmp, "synth_gt.csv")
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
             "--gt", gt, "--results", results, "--out", os.path.join(tmp, "out"), "--no-detail"],
            capture_output=True, text=True)
        print(out.stdout[-1800:])
        assert out.returncode == 0, out.stderr[-1500:]

        acc = float(np.mean(labels[graded] == truth[graded]))
        print("frame accuracy on graded synthetic frames: {:.3f}".format(acc))
        assert acc > 0.80, acc

        def n_spells(lab):
            """Possessions = maximal runs of H or of A; dead ball breaks a run."""
            alive = np.isin(lab, ["H", "A"])
            change = (lab[1:] != lab[:-1]) & alive[1:]
            return int(alive[0]) + int(change.sum())

        raw_n = n_spells(run(raw, meta, Config(switch_penalty=0.0)))
        ours, true = n_spells(labels), n_spells(truth)
        print("possessions: truth {}  decoder {}  no switch penalty {}".format(true, ours, raw_n))
        assert 0.7 * true < ours < 1.5 * true, "decoder possession count is off"
        assert raw_n > 2 * ours, "switch penalty is not suppressing flicker"
    print("OK")


if __name__ == "__main__":
    main()
