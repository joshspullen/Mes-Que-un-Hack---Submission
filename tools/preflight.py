"""Pre-submission safety checks. Reads no match data; it looks at git, file names and counts.

    python tools/preflight.py --team-name <team_name> [--final]

Checks, each printed as PASS / WARN / FAIL (exit code 1 if any FAIL):
  1. git tracks no match data, private ground truth, virtualenv or caches
  2. .gitignore still protects data/
  3. no large tracked files (a stray data export is usually big)
  4. the submission CSV exists, is named <team_name>.csv and passes the official validator
  5. the technical document exists and how many TODOs remain (FAIL with --final)
  6. a pinned requirements-lock.txt exists (FAIL with --final)
  7. the working tree is clean (everything committed)
It cannot check that the repo is public, that it is pushed, or that it regenerates the file; the
checklist in docs/SUBMISSION_CHECKLIST.md covers those, and tools/repro_check.py covers the last.
"""

import argparse
import fnmatch
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN = [
    "data/*",             # only data/.gitkeep is allowed (handled below)
    "*.mp4",
    "*_atd.txt", "*_atd.xml",
    "*_homography.csv",
    "*_smart_tagging.xml",
    "*_possession_gt.csv", "*_possession_gt_meta.json", "*_possession_gt_reference.csv",
    ".venv/*", "venv/*",
    "*.parquet", "*.feather", "*.npy", "*.npz", "*.pkl",
    "cache/*", "features/*",
]
# the two trained models are committed on purpose: made from external matches only (see
# models/MANIFEST.json and tools/train_models.py), they contain no data of the provided game
ALLOWED = {"data/.gitkeep", "models/dead_model.pkl", "models/team_model.pkl"}
BIG_BYTES = 5 * 1024 * 1024

results = []


def report(level, message):
    results.append(level)
    print("{:<5} {}".format(level, message))


def git(*args):
    return subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--csv", help="submission file (default submissions/<team_name>.csv)")
    parser.add_argument("--doc", default=os.path.join("docs", "TECHNICAL_DOCUMENT.md"))
    parser.add_argument("--final", action="store_true",
                        help="treat leftover TODOs and a missing lock file as failures")
    args = parser.parse_args()
    strict = "FAIL" if args.final else "WARN"

    tracked = git("ls-files")
    if tracked.returncode != 0:
        report("FAIL", "not a git repository, or git failed: " + tracked.stderr.strip())
        return 1
    files = [f for f in tracked.stdout.splitlines() if f]

    # 1. forbidden tracked files
    bad = [f for f in files if f not in ALLOWED
           and any(fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(os.path.basename(f), pat)
                   for pat in FORBIDDEN)]
    if bad:
        report("FAIL", "git tracks files that must never be public:")
        for f in bad[:20]:
            print("        " + f)
        print("        fix: git rm --cached <file>, add it to .gitignore, and ask before pushing."
              " If it was ever pushed, tell the team: history keeps it.")
    else:
        report("PASS", "no match data, private ground truth or environment files are tracked")

    # 2. .gitignore protects data/
    try:
        text = open(os.path.join(ROOT, ".gitignore")).read()
    except IOError:
        text = ""
    if "data/*" in text or "/data/" in text:
        report("PASS", ".gitignore protects data/")
    else:
        report("FAIL", ".gitignore has no data/ rule")

    # 3. big tracked files
    big = []
    for f in files:
        p = os.path.join(ROOT, f)
        if os.path.isfile(p) and os.path.getsize(p) > BIG_BYTES:
            big.append((f, os.path.getsize(p)))
    if big:
        report("WARN", "large tracked files (is any of this data?): " +
               ", ".join("{} ({:.1f} MB)".format(f, s / 1e6) for f, s in big))
    else:
        report("PASS", "no tracked file is larger than 5 MB")

    # 4. submission file
    csv = args.csv or os.path.join(ROOT, "submissions", args.team_name + ".csv")
    if not os.path.isfile(csv):
        report("FAIL", "submission file not found: " + csv)
    else:
        if os.path.basename(csv) != args.team_name + ".csv":
            report("FAIL", "file must be named {}.csv, not {}".format(
                args.team_name, os.path.basename(csv)))
        checker = os.path.join(ROOT, "evaluation", "validate_submission.py")
        out = subprocess.run([sys.executable, checker, csv], capture_output=True, text=True)
        first = (out.stdout.strip().splitlines() or ["(no output)"])[0]
        if out.returncode == 0:
            report("PASS", "official validator accepts " + os.path.basename(csv) + "  [" + first + "]")
        else:
            report("FAIL", "validator rejected the file: " + first)

    # 5. technical document
    doc = os.path.join(ROOT, args.doc)
    if not os.path.isfile(doc):
        report("FAIL", "technical document missing: " + args.doc)
    else:
        n = open(doc).read().count("TODO")
        if n:
            report(strict, "{} TODO(s) left in {}".format(n, args.doc))
        else:
            report("PASS", "no TODOs left in " + args.doc)

    # 6. pinned requirements
    if os.path.isfile(os.path.join(ROOT, "requirements-lock.txt")):
        report("PASS", "requirements-lock.txt present")
    else:
        report(strict, "requirements-lock.txt missing (run: pip freeze > requirements-lock.txt)")

    # 7. clean tree
    status = git("status", "--porcelain")
    if status.stdout.strip():
        report("WARN", "uncommitted changes:\n        " +
               "\n        ".join(status.stdout.strip().splitlines()[:15]))
    else:
        report("PASS", "working tree is clean")

    fails = results.count("FAIL")
    print("\n{} FAIL, {} WARN, {} PASS".format(fails, results.count("WARN"), results.count("PASS")))
    print("Not checked here: repo is public, everything is pushed, data folders look empty on"
          " GitHub, the file regenerates (tools/repro_check.py), the form is filled in.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
