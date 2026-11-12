"""A small record of how a submission file was made, saved next to it.

    submissions/<team_name>.manifest.json

Paste its sha256 and settings into the technical document; the organisers can compare the hash
with the file you uploaded. Holds no match data, only aggregate label counts, versions and settings.
"""

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(root, *args):
    try:
        out = subprocess.run(["git", "-C", root] + list(args), capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def build(team_name, csv_path, labels, cfg, hooks, root):
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except ImportError:
        pass
    status = _git(root, "status", "--porcelain")
    return {
        "team_name": team_name,
        "file": os.path.basename(csv_path),
        "sha256": sha256_file(csv_path),
        "rows": int(len(labels)),
        "label_counts": {lab: int(np.sum(labels == lab)) for lab in "HADX"},
        "optional_streams_active": sorted(hooks),
        "config": dataclasses.asdict(cfg),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_tree_dirty": bool(status) if status is not None else None,
        "versions": versions,
        "command": " ".join(sys.argv),
    }


def write(path, manifest):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
