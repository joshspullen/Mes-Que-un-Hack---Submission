"""Train the two models the submission uses, from external matches only, and save them in models/.

    python tools/build_external_cache.py     # once: degrade the 30 external matches (~5 min)
    python tools/train_models.py             # ~3 min; writes models/*.pkl + models/MANIFEST.json
    python tools/train_models.py --check     # retrain and compare with the committed files

Team model (mqh/team_model.py) and dead-ball model (mqh/dead_model.py), each a gradient-boosted
classifier trained on every frame-5th graded frame of the 30 external matches (Metrica 3,
SkillCorner 20, IDSSE 7), degraded with the `bar_ath` and `match_like` profiles. Single thread,
fixed seeds: the same data and the same scikit-learn version give byte-identical files, which
--check verifies. The models are committed so that make_submission.py needs no external download;
this script is how they were made.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
from bench_all import cases, fit_fold  # noqa: E402

MODELS = os.path.join(ROOT, "models")
PROFILES = ("bar_ath", "match_like")


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="retrain into a temp folder and compare")
    args = parser.parse_args()
    import sklearn

    paths = [c[2] for prof in PROFILES for c in cases(prof)]
    if len(paths) < 2 * 30:
        print("warning: {} cached cases, expected 60 (run tools/build_external_cache.py)".format(len(paths)))
    tm, dm = fit_fold(paths)
    out = os.path.join(ROOT, ".cache", "models-check") if args.check else MODELS
    os.makedirs(out, exist_ok=True)
    for name, model in (("team_model.pkl", tm), ("dead_model.pkl", dm)):
        with open(os.path.join(out, name), "wb") as handle:
            pickle.dump(model, handle, protocol=4)
    manifest = {"sklearn": sklearn.__version__, "training_cases": len(paths), "profiles": list(PROFILES),
                "files": {n: sha(os.path.join(out, n)) for n in ("team_model.pkl", "dead_model.pkl")}}
    with open(os.path.join(out, "MANIFEST.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))
    if args.check:
        with open(os.path.join(MODELS, "MANIFEST.json")) as handle:
            ref = json.load(handle)
        same = ref["files"] == manifest["files"]
        print("identical to the committed models:", same)
        sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
