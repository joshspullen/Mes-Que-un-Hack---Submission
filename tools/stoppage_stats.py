"""How often does possession change team across a stoppage? Measured on EXTERNAL open data only.

Source: Metrica Sports sample data, games 1 and 2 (anonymised event data, public repository
https://github.com/metrica-sports/sample-data). Its README asks that public use acknowledges the
source: this is that acknowledgement. Nothing from the hackathon match is read here, and the
numbers only inform the *prior* behind two decoder constants (restart_switch_scale and the
handover placement); they are never fitted on BAR-ATH.

    python tools/stoppage_stats.py                 # downloads the two CSVs at run time
    python tools/stoppage_stats.py --dir some/dir  # or reads game1.csv / game2.csv from a folder

Prints aggregate rates only. For every restart (SET PIECE) it compares the restarting team with the
team that last had the ball in open play (last PASS / SHOT / RECOVERY before the restart).
"""

import argparse
import csv
import io
import os
import urllib.request
from collections import defaultdict

URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
       "Sample_Game_{n}/Sample_Game_{n}_RawEventsData.csv")
HOLDING = ("PASS", "SHOT", "RECOVERY")
FPS = 25.0


def load(n, folder):
    if folder:
        with open(os.path.join(folder, "game{}.csv".format(n)), newline="") as handle:
            return list(csv.DictReader(handle))
    with urllib.request.urlopen(URL.format(n=n), timeout=60) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))


def analyse(rows):
    """[(subtype, changed_team, dead_seconds)] for each restart that follows open play."""
    out = []
    last_team, last_end = None, None
    for r in rows:
        kind = r["Type"]
        if kind in HOLDING:
            last_team = r["Team"]
            last_end = float(r["End Time [s]"] or r["Start Time [s]"])
        elif kind == "SET PIECE" and r["Subtype"] != "KICK OFF" and last_team:
            start = float(r["Start Time [s]"])
            out.append((r["Subtype"], r["Team"] != last_team, max(0.0, start - last_end)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", help="folder with game1.csv and game2.csv (otherwise downloaded)")
    args = parser.parse_args()

    by_subtype = defaultdict(list)
    for n in (1, 2):
        for subtype, changed, gap in analyse(load(n, args.dir)):
            by_subtype[subtype].append((changed, gap))
            by_subtype["ALL RESTARTS"].append((changed, gap))

    print("{:<14}{:>6}{:>16}{:>18}".format("restart", "n", "team changed", "median gap (s)"))
    for subtype in sorted(by_subtype):
        items = by_subtype[subtype]
        changed = sum(c for c, _ in items) / len(items)
        gaps = sorted(g for _, g in items)
        print("{:<14}{:>6}{:>15.0%}{:>18.1f}".format(subtype, len(items), changed, gaps[len(gaps) // 2]))
    print("\n'gap' = last open-play event end -> restart start; it includes the ball travelling out and any"
          " silent events, so it is an upper bound on the dead time.")


if __name__ == "__main__":
    main()
