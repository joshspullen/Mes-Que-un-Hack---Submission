"""Degrade every external match and cache what the models and the benches need.

    python tools/build_external_cache.py                       # all providers, profiles bar_ath + match_like
    python tools/build_external_cache.py --providers metrica   # a subset

For each (match, profile, seed): the degraded tracking goes through exactly the code the submission
runs (mqh.pipeline.prepare_arrays with the gap and touch hooks; Smart Tagging hooks skip themselves
on external matches), then the feature tables of the team model (mqh/team_model.py) and the
dead-ball model (mqh/dead_model.py) are computed. Saved to .cache/external/case-*.npz (git-ignored).
Nothing here reads BAR-ATH.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mqh.external import registry  # noqa: E402


def case_path(provider, match, profile, seed):
    return os.path.join(registry.CACHE, "case-{}-{}-{}-s{}.npz".format(provider, match, profile, seed))


def build(args):
    provider, match, profile, seed = args
    path = case_path(provider, match, profile, seed)
    if os.path.exists(path):
        return path, 0.0
    from mqh import dead_model, team_model
    from mqh.dead import tracking_dead_mask
    from mqh.pipeline import Config, discover_hooks, prepare_arrays
    from mqh.external.degrade import PROFILES

    t0 = time.time()
    g = registry.load(provider, match)
    pitch, fps = (105.0, 68.0), 25
    ball, players = registry.degraded(g, profile, seed)
    hooks = discover_hooks()
    cfg = Config()
    evidence, dead, blocks, n, touches = prepare_arrays(ball, players, g["team_of_slot"], cfg, pitch, fps,
                                                        g["blocks"], hooks=hooks)
    anti, sym = team_model.team_features(ball, players, g["team_of_slot"], pitch, fps, evidence)
    F = dead_model.features(ball, players, g["team_of_slot"], pitch, fps)
    tf, tt = touches if touches is not None else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    np.savez_compressed(path, anti=anti.astype(np.float32), sym=sym.astype(np.float32), deadF=F,
                        evidence=evidence.astype(np.float32), dead_v2=dead,
                        blocks=np.array([[b[1], b[2]] for b in blocks]), touch_frames=tf, touch_teams=tt,
                        **{"truth_" + k: v for k, v in g["truths"].items()})
    return path, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--providers", nargs="+", default=list(registry.PROVIDERS))
    parser.add_argument("--profiles", nargs="+", default=["bar_ath", "match_like"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    jobs = [(p, m, prof, s) for p, m in registry.all_matches(tuple(args.providers))
            for prof in args.profiles for s in args.seeds]
    # load (and cache) the clean matches once, serially: the XML parsing is memory hungry
    for p, m in sorted({(j[0], j[1]) for j in jobs}):
        registry.load(p, m)
    with Pool(args.jobs) as pool:
        for path, dt in pool.imap_unordered(build, jobs):
            print("{:6.1f}s {}".format(dt, os.path.basename(path)), flush=True)


if __name__ == "__main__":
    main()
