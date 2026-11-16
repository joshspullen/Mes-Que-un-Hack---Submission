"""Before/after for the team_evidence changes, on the real match.

    python tests/compare_evidence.py
    python evaluation/run_evaluation.py --bootstrap 1000

Writes two submissions into evaluation/results/ (git-ignored):
  evidence_before.csv  -- the new terms switched off: nearest-player margin only (= v0 evidence)
  evidence_after.csv   -- speed weighting + nearby-player count + contested balls
Dead ball, decoder and switch penalty come from the stream 5 pipeline and are identical for both;
the optional stream hooks are switched off (hooks={}), so any difference comes from team_evidence
alone. The public 5 minutes are a sanity check, not a tuning set (RULES.md).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402
from mqh.atd import load_atd  # noqa: E402
from mqh.decode import EvidenceConfig, team_evidence  # noqa: E402
from mqh.pipeline import Config, decode, prepare, write_csv  # noqa: E402

BEFORE = EvidenceConfig(margin_share=1.0, speed_mid_ms=1e9, contest_floor=1.0)
AFTER = EvidenceConfig()


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config()
    # dead mask and half blocks from the pipeline; evidence recomputed with each EvidenceConfig
    _, dead, blocks, n_frames = prepare(raw_path, meta_path, cfg, hooks={})
    ball, players, team_of_slot = load_atd(raw_path, meta_path, n_frames)
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]
    labels = {}
    for name, ecfg in (("evidence_before", BEFORE), ("evidence_after", AFTER)):
        evidence = team_evidence(ball, players, team_of_slot, pitch, ecfg, fps=fps)
        labels[name] = decode(evidence, dead, blocks, n_frames, cfg.decoder)
        path = os.path.join(out_dir, name + ".csv")
        write_csv(path, labels[name])
        counts = "  ".join("{}={}".format(c, int(np.sum(labels[name] == c))) for c in "HADX")
        print("wrote {}   {}".format(path, counts))
    a, b = labels["evidence_before"], labels["evidence_after"]
    print("frames that changed label: {} ({:.1%})".format(int(np.sum(a != b)), np.mean(a != b)))


if __name__ == "__main__":
    main()
