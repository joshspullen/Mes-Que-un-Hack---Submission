"""SkillCorner open data (20 A-League 2024/25 matches, MIT licence) -> our arrays + H/A/D labels.

Source: https://github.com/SkillCorner/opendata (tracking files are Git LFS objects; fetched from
media.githubusercontent.com by tools/external_bench.py into data/external/, git-ignored).

Broadcast tracking at 10 Hz, players out of the camera view extrapolated, ball often undetected:
a different provider and a harder view than Metrica's, which is the point (more variety for the
models, and a check that choices do not depend on one provider's habits). Resampled to 25 fps.

Labels, by code from SkillCorner's own "phases of play" (`*_phases_of_play.csv`): a phase is a
spell of one team's possession in play; phases follow each other without a gap while the ball is
in play. So:
  H / A  inside a phase, the phase's `team_in_possession_id`
  D      a gap of more than 2 s between consecutive phases (a stoppage), and from the period start
         to the first phase; gaps up to 2 s are bridged with the previous phase's team
This is SkillCorner's team-possession convention (analyst-reviewed), close to "controlled
possession"; a third reading next to Metrica's strict / loose, which keeps choices honest.
"""

import csv
import json
import os

import numpy as np

FPS_SRC = 10
FPS = 25
PITCH = (105.0, 68.0)
HOME, AWAY = 0, 1
BRIDGE_S = 2.0


def match_ids(root):
    d = os.path.join(root, "data", "matches")
    return sorted(m for m in os.listdir(d) if os.path.isdir(os.path.join(d, m)))


def load_game(match_id, root):
    d = os.path.join(root, "data", "matches", str(match_id))
    with open(os.path.join(d, "{}_match.json".format(match_id))) as handle:
        meta = json.load(handle)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    slot_of, teams = {}, []
    for p in meta["players"]:
        slot_of[p["id"]] = len(teams)
        teams.append(HOME if p["team_id"] == home_id else AWAY)

    rows = []
    with open(os.path.join(d, "{}_tracking_extrapolated.jsonl".format(match_id))) as handle:
        for line in handle:
            rows.append(json.loads(line))
    n_src = rows[-1]["frame"] + 1
    ball = np.full((n_src, 2), np.nan)
    players = np.full((n_src, len(teams), 2), np.nan)
    period = np.zeros(n_src, dtype=np.int8)
    for r in rows:
        f = r["frame"]
        period[f] = r["period"] or 0
        b = r["ball_data"]
        # an undetected ball is SkillCorner's estimate; keep only detected positions, like a tracker
        if b["x"] is not None and b.get("is_detected"):
            ball[f] = (b["x"], b["y"])
        for p in r["player_data"]:
            j = slot_of.get(p["player_id"])
            if j is not None:
                players[f, j] = (p["x"], p["y"])

    # centred metres -> normalised [0, 1], y downwards like the ATD
    def norm(xy):
        out = np.empty_like(xy)
        out[..., 0] = xy[..., 0] / length + 0.5
        out[..., 1] = 0.5 - xy[..., 1] / width
        return out

    ball, players = norm(ball), norm(players)

    # 10 Hz -> 25 fps: linear interpolation between valid neighbours, NaN across gaps
    n = int(np.floor((n_src - 1) * FPS / FPS_SRC)) + 1
    t = np.arange(n) * FPS_SRC / FPS
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, n_src - 1)
    w = (t - i0)[:, None]
    ball25 = ball[i0] * (1 - w) + ball[i1] * w
    players25 = players[i0] * (1 - w[:, None]) + players[i1] * w[:, None]
    period25 = period[np.rint(t).astype(int).clip(0, n_src - 1)]

    phases = []
    with open(os.path.join(d, "{}_phases_of_play.csv".format(match_id)), newline="") as handle:
        for r in csv.DictReader(handle):
            phases.append((int(r["period"]), int(r["frame_start"]), int(r["frame_end"]),
                           HOME if int(r["team_in_possession_id"]) == home_id else AWAY))
    periods = [(p["period"], p["start_frame"], p["end_frame"]) for p in meta["match_periods"]]
    return {"name": "sc{}".format(match_id), "ball": ball25, "players": players25,
            "team_of_slot": np.array(teams, dtype=np.int8), "period": period25,
            "phases": phases, "periods": periods}


def _to25(f):
    return int(round(f * FPS / FPS_SRC))


def blocks_of(game):
    """[(period, first, last)]: from the period's first phase (the kick-off) to its end, at 25 fps."""
    out = []
    n = len(game["ball"])
    for p, _, end in game["periods"]:
        ph = [x for x in game["phases"] if x[0] == p]
        if not ph:
            continue
        out.append((p, _to25(ph[0][1]), min(_to25(end), n - 1)))
    return out


def labels(game, convention=None):
    """Per-frame 'H'/'A'/'D'/'X' at 25 fps (convention ignored: SkillCorner has one reading)."""
    n = len(game["ball"])
    truth = np.full(n, "X", dtype="<U1")
    bridge = BRIDGE_S * FPS_SRC
    for p, first, last in blocks_of(game):
        lab = np.full(last - first + 1, "D", dtype="<U1")
        ph = sorted(x for x in game["phases"] if x[0] == p)
        for k, (_, a, b, team) in enumerate(ph):
            end = b
            if k + 1 < len(ph) and ph[k + 1][1] - b <= bridge:
                end = ph[k + 1][1]
            lo, hi = max(_to25(a), first), min(_to25(end), last + 1)
            if lo < hi:
                lab[lo - first:hi - first] = "H" if team == HOME else "A"
        truth[first:last + 1] = lab
    return truth
