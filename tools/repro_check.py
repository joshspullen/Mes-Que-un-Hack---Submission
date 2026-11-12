"""Does the repo regenerate the same submission file every time?

    python tools/repro_check.py [--runs 2]

Runs make_submission.py several times into separate temp folders and compares SHA-256 hashes.
Identical hashes mean the pipeline is deterministic. Put the hash in your technical document so
the organisers can compare it with the file you uploaded. Needs the real data in ./data (or
$MTAH_DATA_DIR); prints only the hash and pass/fail, no match values.

For the full proof, do it on a fresh clone: git clone <repo> /tmp/fresh && cd /tmp/fresh,
copy the data into its data/tactical/, install requirements, and run this script there.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--team-name", default="repro_check")
    args = parser.parse_args()

    hashes = []
    for i in range(args.runs):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, os.path.join(ROOT, "make_submission.py"),
                 "--team-name", args.team_name, "--out-dir", tmp],
                capture_output=True, text=True)
            out = os.path.join(tmp, args.team_name + ".csv")
            if proc.returncode != 0 or not os.path.isfile(out):
                print("FAIL  run {} did not produce a valid file:\n{}".format(
                    i + 1, (proc.stdout + proc.stderr)[-800:]))
                return 1
            hashes.append(sha256(out))
            print("run {}  sha256 {}".format(i + 1, hashes[-1]))

    if len(set(hashes)) == 1:
        print("PASS  identical output on {} runs".format(args.runs))
        return 0
    print("FAIL  outputs differ between runs: something in the pipeline is not deterministic")
    return 1


if __name__ == "__main__":
    sys.exit(main())
