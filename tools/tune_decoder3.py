"""Choose the decoder on out-of-fold predictions of the 30 external matches.

    python tools/tune_decoder3.py [profile]      # after tools/bench_all.py

Compares the two-step decoder (threshold the dead-ball probability, then decode the team) with the
joint H/A/D decoder (mqh/decoder3.py) over a grid of costs. Each external match is scored with the
probabilities of models that never saw it (out-of-fold), so the comparison is fair. The score is
the mean spellF1 per provider, each provider weighing the same; robustness check: a setting chosen
on two providers is scored on the third.
"""

import glob
import itertools
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evaluation.possession_metrics as pm  # noqa: E402
from mqh.decoder import DecoderConfig  # noqa: E402
from mqh.decoder3 import Decoder3Config, decode3  # noqa: E402
from mqh.external import registry  # noqa: E402
from mqh.pipeline import decode  # noqa: E402


def load(profile):
    out = []
    for path in sorted(glob.glob(os.path.join(registry.CACHE, "oof-*-{}-s0.npz".format(profile)))):
        provider = os.path.basename(path).split("-")[1]
        o = np.load(path)
        c = np.load(path.replace("oof-", "case-"))
        conv = "truth_" + registry.PROVIDERS[provider]["conventions"][0]
        blocks = [(i + 1, int(a), int(b)) for i, (a, b) in enumerate(c["blocks"])]
        out.append(dict(provider=provider, p_home=o["p_home"].astype(float), p_dead=o["p_dead"].astype(float),
                        truth=c[conv], blocks=blocks, touches=(c["touch_frames"], c["touch_teams"])))
    return out


def spell(truth, labels):
    return pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))["spell_f1_mean"]


def per_provider(cases, fn):
    by = {}
    for c in cases:
        by.setdefault(c["provider"], []).append(spell(c["truth"], fn(c)))
    return {p: float(np.mean(v)) for p, v in by.items()}


def main(profile="bar_ath"):
    cases = load(profile)
    print("{} cases, profile {}".format(len(cases), profile))
    rows = []

    def run(name, fn):
        r = per_provider(cases, fn)
        rows.append((name, r))
        print("{:<58} {}  mean {:.4f}".format(name, "  ".join("{} {:.3f}".format(k[:5], v) for k, v in r.items()),
                                              np.mean(list(r.values()))), flush=True)

    for th, pen in itertools.product((0.4, 0.5), (5.0, 10.0, 15.0, 20.0)):
        dc = replace(DecoderConfig(), switch_penalty=pen)
        run("two-step  dead>={} penalty {}".format(th, pen),
            lambda c, th=th, dc=dc: decode(2 * c["p_home"] - 1, c["p_dead"] >= th, c["blocks"],
                                           len(c["truth"]), dc, c["touches"]))
    for tc, sc, rc in itertools.product((5.0, 10.0, 20.0), (2.0, 5.0, 10.0), (2.0, 5.0, 10.0)):
        cfg = Decoder3Config(team_cost=tc, stop_cost=sc, restart_cost=rc)
        run("joint team {} stop {} restart {}".format(tc, sc, rc),
            lambda c, cfg=cfg: decode3(c["p_home"], c["p_dead"], c["blocks"], len(c["truth"]), cfg))

    provs = sorted({c["provider"] for c in cases})
    print("\nrobustness: chosen on two providers, scored on the third")
    for p in provs:
        others = [q for q in provs if q != p]
        best = max(rows, key=lambda r: np.mean([r[1][q] for q in others]))
        print("  held-out {:<12} chosen '{}' -> {:.3f} (best possible {:.3f})".format(
            p, best[0], best[1][p], max(r[1][p] for r in rows)))
    best = max(rows, key=lambda r: np.mean(list(r[1].values())))
    print("\nbest overall: {}  {:.4f}".format(best[0], np.mean(list(best[1].values()))))


if __name__ == "__main__":
    main(*sys.argv[1:])
