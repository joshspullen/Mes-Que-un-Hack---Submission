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
from mqh.decoder import DecoderConfig  # noqa: E402
from mqh.pipeline import Config, discover_hooks, run, write_csv  # noqa: E402
import evaluation.possession_metrics as pm  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        truth = synth.write(tmp)
        raw, meta = os.path.join(tmp, "synth_atd.txt"), os.path.join(tmp, "synth_atd.xml")
        labels = run(raw, meta, Config(), hooks={})

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
                  run(raw, meta, Config(decoder=DecoderConfig(switch_penalty=0.0)), hooks={}))

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

        raw_n = n_spells(run(raw, meta, Config(decoder=DecoderConfig(switch_penalty=0.0)), hooks={}))
        ours, true = n_spells(labels), n_spells(truth)
        print("possessions: truth {}  decoder {}  no switch penalty {}".format(true, ours, raw_n))
        assert 0.7 * true < ours < 1.5 * true, "decoder possession count is off"
        assert raw_n > 2 * ours, "switch penalty is not suppressing flicker"

        # --- hooks: the plug-in points for streams 1 and 4 (docs/INTERFACES.md) ---
        n = synth.N
        # The synthetic world has no Smart Tagging file, so these tests switch auto-discovery off
        # (hooks={}) and inject fakes. Streams 1 and 4 test their own modules.
        none = labels
        assert set(discover_hooks()) <= {"restarts", "gaps", "touches"}, "unknown hook discovered"

        zero_gaps = run(raw, meta, Config(), hooks={"gaps": lambda *a: np.zeros(n)})
        assert (zero_gaps == none).all(), "a gap hook with no opinion must change nothing"

        truth_dead = truth == "D"
        # cleaning off here: it (rightly) drops dead runs shorter than 1 s, and the synthetic match
        # has a few truncated ones at the end of a half; this check is only about the plumbing
        with_restarts = run(raw, meta, Config(decoder=DecoderConfig(clean_dead_mask=False)),
                            hooks={"restarts": lambda n_frames, game: truth_dead.copy()})
        assert (with_restarts[truth_dead] == "D").all(), "restart hook mask was not applied"
        print("hook plumbing ok (dead frames forced to D: {})".format(int(truth_dead.sum())))

        for name, hook, key in [
                ("restart_dead_mask", lambda n_frames, game: np.zeros(5, dtype=bool), "restarts"),
                ("restart_dead_mask", lambda n_frames, game: np.zeros(n_frames), "restarts"),
                ("gap_evidence", lambda *a: np.full(n, np.nan), "gaps")]:
            try:
                run(raw, meta, Config(), hooks={key: hook})
            except ValueError as error:
                assert name in str(error), str(error)
            else:
                raise AssertionError("bad {} hook was accepted".format(name))
        print("bad hooks are rejected with the hook's name")

        # restart-team evidence: tell the decoder which team restarts, from the (synthetic) truth
        push = np.zeros(n)
        for i in range(1, n):
            if truth[i - 1] == "D" and truth[i] in ("H", "A"):
                push[i:i + 50] = 0.8 if truth[i] == "H" else -0.8
        with_team = run(raw, meta, Config(), hooks={"restart_team": lambda n_frames, game: push})
        acc_with = float(np.mean(with_team[graded] == truth[graded]))
        print("restart-team evidence: accuracy {:.3f} -> {:.3f}".format(acc, acc_with))
        assert acc_with >= acc, "restart-team evidence made things worse on a world where it is exact"
        try:
            run(raw, meta, Config(), hooks={"restart_team": lambda n_frames, game: push * 5})
        except ValueError as error:
            assert "restart_evidence" in str(error), str(error)
        else:
            raise AssertionError("out-of-range restart_evidence was accepted")

        # touches: the plug-in for boundary snapping (off by default, so a new hook cannot silently
        # change the submission). Fake touches at the true possession starts.
        t_frames = np.array([i for i in range(1, n) if truth[i] in "HA" and truth[i] != truth[i - 1]])
        t_teams = np.array([0 if truth[i] == "H" else 1 for i in t_frames])
        touch_hook = lambda *a: (t_frames, t_teams)   # noqa: E731
        off = run(raw, meta, Config(), hooks={"touches": touch_hook})
        assert (off == labels).all(), "touches must change nothing while snap_window_frames == 0"
        on = run(raw, meta, Config(decoder=DecoderConfig(snap_window_frames=25)),
                 hooks={"touches": touch_hook})
        assert on.shape == labels.shape and set(np.unique(on)) <= set("HADX")
        assert (on[~graded] == "X").all() and (on[graded] != "X").all()
        print("touches hook ok: {} frames moved when enabled".format(int(np.sum(on != labels))))
        for bad in [lambda *a: np.zeros(5), lambda *a: (np.array([0.5]), np.array([0])),
                    lambda *a: (np.array([n + 5]), np.array([0])),
                    lambda *a: (np.array([5]), np.array([2])),
                    lambda *a: (np.array([5, 6]), np.array([0]))]:
            try:
                run(raw, meta, Config(), hooks={"touches": bad})
            except ValueError as error:
                assert "touch_events" in str(error), str(error)
            else:
                raise AssertionError("bad touch_events hook was accepted")
    print("OK")


if __name__ == "__main__":
    main()
