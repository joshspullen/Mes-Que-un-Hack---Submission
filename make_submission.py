"""One command from the provided data to the submission file.

    python make_submission.py --team-name <team_name>

Reads data/tactical/BAR-ATH_tactical_atd.{xml,txt} (or $MTAH_DATA_DIR), writes
submissions/<team_name>.csv, then validates it with the official checker.
Deterministic: no randomness anywhere in the pipeline.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import require  # noqa: E402
from mqh import manifest  # noqa: E402
from mqh.decoder import DecoderConfig  # noqa: E402
from mqh.pipeline import Config, discover_hooks, run, write_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team-name", required=True, help="registered team name (names the file)")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "submissions"))
    parser.add_argument("--switch-penalty", type=float, default=DecoderConfig.switch_penalty)
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    cfg = Config(decoder=DecoderConfig(switch_penalty=args.switch_penalty))
    hooks = discover_hooks()
    print("optional streams active:", ", ".join(sorted(hooks)) or "none")
    labels = run(raw_path, meta_path, cfg, hooks=hooks)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.team_name + ".csv")
    write_csv(out, labels)
    print("wrote", out)
    info = manifest.build(args.team_name, out, labels, cfg, hooks, HERE)
    print("manifest", manifest.write(os.path.join(args.out_dir, args.team_name + ".manifest.json"), info))
    print("sha256", info["sha256"])
    for lab in "HADX":
        print("  {} {:>7}".format(lab, int(np.sum(labels == lab))))

    checker = os.path.join(HERE, "evaluation", "validate_submission.py")
    return subprocess.call([sys.executable, checker, out])


if __name__ == "__main__":
    sys.exit(main())
