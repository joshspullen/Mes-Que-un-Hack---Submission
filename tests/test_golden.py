"""Did my change alter the labels? A fixed made-up match must give a fixed hash.

    python tests/test_golden.py            # compare with tests/golden.json
    python tests/test_golden.py --update   # after an INTENDED change, store the new hash

With four people merging into one pipeline, this answers "did that merge change the output?"
immediately. A mismatch is not a bug by itself: it means the labels changed, so confirm that was
the intention, then run with --update and mention the change in the pull request. The hash also
checks that the decoder is reproducible across machines (it works in integers).
Uses a made-up match with turnovers, so it exercises restarts and handovers inside ball gaps.
"""

import dataclasses
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import synth  # noqa: E402
from mqh.pipeline import Config, run  # noqa: E402

GOLDEN = os.path.join(HERE, "golden.json")


def current():
    with tempfile.TemporaryDirectory() as tmp:
        synth.write(tmp, turnover_frac=0.4, seed=11, dead_flip_prob=0.5)
        labels = run(os.path.join(tmp, "synth_atd.txt"), os.path.join(tmp, "synth_atd.xml"),
                     Config(), hooks={})
    return {"labels_sha256": hashlib.sha256("".join(labels.tolist()).encode()).hexdigest(),
            "world": "synth turnover=0.4 seed=11 dead_flip=0.5",
            "config": dataclasses.asdict(Config())}


def main():
    now = current()
    if "--update" in sys.argv:
        with open(GOLDEN, "w") as f:
            json.dump(now, f, indent=2, sort_keys=True)
            f.write("\n")
        print("golden hash updated:", now["labels_sha256"])
        return 0
    with open(GOLDEN) as f:
        want = json.load(f)
    if now["labels_sha256"] != want["labels_sha256"]:
        print("MISMATCH: the labels for the fixed synthetic match changed.")
        print("  stored  ", want["labels_sha256"])
        print("  current ", now["labels_sha256"])
        if now["config"] != want["config"]:
            print("  (the default Config also changed, which would explain it)")
        print("If this change is intended, run: python tests/test_golden.py --update")
        return 1
    print("OK golden hash unchanged", now["labels_sha256"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
