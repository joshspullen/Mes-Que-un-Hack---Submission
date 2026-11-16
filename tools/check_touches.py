"""Check stream 3b (mqh/touches.py) against EXTERNAL event data: Metrica's sample games.

    python tools/check_touches.py

A detected touch is correct if a Metrica event of the same team (PASS / SHOT start, PASS end =
reception, CARRY / RECOVERY / CHALLENGE / BALL LOST start) lies within +-5 frames. Recall is measured
on pass starts and receptions while the ball is tracked. Degraded copies, as in external_bench.py.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
from external_bench import DEFAULT_DIR, build_inputs, fetch  # noqa: E402
from mqh.external import metrica  # noqa: E402
from mqh.touches import touch_events  # noqa: E402

TOL = 5


def main():
    games = {g["name"]: g for g in (metrica.load_game(n, fetch(DEFAULT_DIR)) for n in (1, 2, 3))}
    for name, ball, players, team_of_slot, _, _ in build_inputs(list(games.values()), [0]):
        g = games[name.split("-")[0]]
        frames, teams = touch_events(ball, players, team_of_slot, metrica.PITCH, metrica.FPS)
        ref = []
        for e in g["events"]:
            if e["type"] in ("PASS", "SHOT", "CARRY", "RECOVERY", "CHALLENGE", "BALL LOST", "SET PIECE"):
                ref.append((e["start"], e["team"]))
            if e["type"] == "PASS" and e["end"] > e["start"]:
                ref.append((e["end"], e["team"]))
        ref = np.array(ref)
        seen = ~np.isnan(ball[:, 0])
        ok = 0
        for f, t in zip(frames, teams):
            near = np.abs(ref[:, 0] - f) <= TOL
            ok += bool(near.any() and (ref[near, 1] == t).any())
        targets = [(e["start"], e["team"]) for e in g["events"] if e["type"] == "PASS"] + \
                  [(e["end"], e["team"]) for e in g["events"] if e["type"] == "PASS" and e["end"] > e["start"]]
        targets = [(f, t) for f, t in targets if 0 <= f < len(seen) and seen[f]]
        hit = sum(bool(((np.abs(frames - f) <= TOL) & (teams == t)).any()) for f, t in targets)
        print("{:<12} touches {:>5}  precision {:.2f}  recall (passes, ball seen) {:.2f}  [{} targets]".format(
            name, len(frames), ok / max(1, len(frames)), hit / max(1, len(targets)), len(targets)))


if __name__ == "__main__":
    main()
