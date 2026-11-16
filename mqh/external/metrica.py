"""Metrica Sports sample data (3 anonymised games) -> the same arrays as our ATD, plus H/A/D labels.

Source: https://github.com/metrica-sports/sample-data (README: free to use, please acknowledge the
source). Downloaded at run time by tools/external_bench.py into data/external/ (git-ignored).

Arrays, exactly the convention of mqh/atd.py so every stream runs unchanged on them:
  ball         (n, 2)        normalised x, y, NaN where the ball is not tracked
  players      (n, slots, 2) normalised x, y, NaN where a slot is empty
  team_of_slot (slots,)      0 = home, 1 = away
Video frame = Metrica frame - 1. 25 fps, pitch 105 x 68 m.

Labels are derived from Metrica's (human) event data BY CODE, with fixed rules written below. They
approximate, and do not equal, the organisers' convention (corrected tracking). Two readings of
"controlled possession" are produced so that a choice that only works under one of them shows up:

  loose  -- the team of every PASS / CARRY / SHOT / SET PIECE / RECOVERY / BALL LOST event owns the
            ball from that event's first frame (a one-touch interception counts as a possession).
  strict -- a change of team needs control: a RECOVERY, a CARRY or a SET PIECE by the new team, or
            two consecutive controlled events by it. A single touch by the other team (a first-time
            clearance that goes straight back) does not change possession (CHALLENGE.md).

Dead ball: from a stoppage trigger (BALL OUT, a SHOT that goes out or scores, a FAULT RECEIVED or a
foul in a CHALLENGE, an offside / hand ball BALL LOST) to the next restart (SET PIECE, or a goal
kick PASS). If open play continues before any restart (advantage), the trigger is ignored.
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

PITCH = (105.0, 68.0)
FPS = 25
HOME, AWAY = 0, 1

CONTROLLED = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST"}
STRONG_CONTROL = {"RECOVERY", "CARRY", "SET PIECE"}
OPEN_PLAY = {"PASS", "CARRY", "SHOT", "RECOVERY"}


# ---------------------------------------------------------------------------------------------
# events

def _subtype_text(sub):
    if sub is None:
        return ""
    if isinstance(sub, dict):
        return sub.get("name", "") or ""
    if isinstance(sub, list):
        return "-".join(s.get("name", "") for s in sub if isinstance(s, dict))
    return str(sub)


def _events_csv(path):
    out = []
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            start = int(float(r["Start Frame"])) - 1
            end = int(float(r["End Frame"] or 0)) - 1
            out.append({"team": HOME if r["Team"] == "Home" else AWAY, "type": r["Type"],
                        "subtype": r["Subtype"] or "", "period": int(r["Period"]),
                        "start": start, "end": max(end, start)})
    return out


def _events_json(path, home_id):
    out = []
    with open(path) as handle:
        data = json.load(handle)["data"]
    for e in data:
        start = int(e["start"]["frame"]) - 1
        end = int(e["end"]["frame"] or e["start"]["frame"]) - 1
        out.append({"team": HOME if e["team"]["id"] == home_id else AWAY, "type": e["type"]["name"],
                    "subtype": _subtype_text(e.get("subtypes")), "period": int(e["period"]),
                    "start": start, "end": max(end, start)})
    return out


# ---------------------------------------------------------------------------------------------
# tracking

def _tracking_csv(home_path, away_path):
    def read(path):
        with open(path) as handle:
            rows = handle.read().splitlines()[3:]
        arr = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in row.split(",")]
                        for row in rows])
        return arr

    home, away = read(home_path), read(away_path)
    n = int(max(home[:, 1].max(), away[:, 1].max()))
    period = np.zeros(n, dtype=np.int8)
    period[home[:, 1].astype(int) - 1] = home[:, 0].astype(np.int8)
    ball = np.full((n, 2), np.nan)
    ball[home[:, 1].astype(int) - 1] = home[:, -2:]
    blocks, teams = [], []
    for arr, team in ((home, HOME), (away, AWAY)):
        xy = arr[:, 3:-2]
        k = xy.shape[1] // 2
        p = np.full((n, k, 2), np.nan)
        p[arr[:, 1].astype(int) - 1] = xy.reshape(len(arr), k, 2)
        blocks.append(p)
        teams += [team] * k
    return ball, np.concatenate(blocks, axis=1), np.array(teams, dtype=np.int8), period


def _tracking_epts(raw_path, meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    player_team = {p.get("id"): (HOME if p.get("teamId") == home_id else AWAY)
                   for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    players_all = list(player_team)
    slot_of = {pid: i for i, pid in enumerate(players_all)}

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [slot_of[channel_player[r]] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))

    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    h2 = int(params.get("second_half_start", 0) or 0)

    with open(raw_path) as handle:
        lines = handle.read().splitlines()
    n = len(lines)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, len(players_all), 2), np.nan)
    period = np.ones(n, dtype=np.int8)
    for line in lines:
        frame_s, body, ball_s = line.split(":")
        f = int(frame_s)
        order = next(o for a, b, o in specs if a <= f <= b)
        vals = np.fromstring(body.replace(";", ","), sep=",")
        if vals.size == 2 * len(order):
            players[f - 1, order] = vals.reshape(-1, 2)
        b = ball_s.split(",")
        try:
            ball[f - 1] = (float(b[0]), float(b[1]))
        except (ValueError, IndexError):
            pass
        if h2 and f >= h2:
            period[f - 1] = 2
    team_of_slot = np.array([player_team[p] for p in players_all], dtype=np.int8)
    return ball, players, team_of_slot, period, home_id


def load_game(n, root):
    """dict(name, ball, players, team_of_slot, period, events) for Metrica sample game n (1-3)."""
    d = os.path.join(root, "data", "Sample_Game_{}".format(n))
    if n in (1, 2):
        stem = os.path.join(d, "Sample_Game_{}_".format(n))
        ball, players, team_of_slot, period = _tracking_csv(
            stem + "RawTrackingData_Home_Team.csv", stem + "RawTrackingData_Away_Team.csv")
        events = _events_csv(stem + "RawEventsData.csv")
    else:
        ball, players, team_of_slot, period, home_id = _tracking_epts(
            os.path.join(d, "Sample_Game_3_tracking.txt"), os.path.join(d, "Sample_Game_3_metadata.xml"))
        events = _events_json(os.path.join(d, "Sample_Game_3_events.json"), home_id)
    events.sort(key=lambda e: (e["start"], e["end"]))
    return {"name": "metrica{}".format(n), "ball": ball, "players": players,
            "team_of_slot": team_of_slot, "period": period, "events": events}


# ---------------------------------------------------------------------------------------------
# labels

def _is_trigger(e):
    t, s = e["type"], e["subtype"].upper()
    if t == "BALL OUT":
        return True
    if t == "SHOT" and ("OUT" in s or "GOAL" in s) and "SAVED" not in s:
        return True
    if t == "FAULT RECEIVED":
        return True
    if t == "CHALLENGE" and "FAULT" in s:
        return True
    if t == "BALL LOST" and any(k in s for k in ("OFFSIDE", "HAND BALL", "END HALF")):
        return True
    return False


def _is_restart(e):
    return e["type"] == "SET PIECE" or ("GOAL KICK" in e["subtype"].upper()
                                         and e["type"] in ("PASS", "BALL LOST"))


def blocks_of(game):
    """[(period, first, last)] graded like BAR-ATH: from the period's kick-off to its last frame."""
    out = []
    for p in (1, 2):
        frames = np.flatnonzero(game["period"] == p)
        kick = [e["start"] for e in game["events"]
                if e["period"] == p and e["type"] == "SET PIECE" and "KICK OFF" in e["subtype"].upper()]
        if frames.size == 0 or not kick:
            continue
        last_event = max(e["end"] for e in game["events"] if e["period"] == p)
        out.append((p, int(min(kick)), int(min(frames.max(), last_event + 2 * FPS))))
    return out


def dead_intervals(events):
    """[(start, end_exclusive)] of stoppages, from the event rules in the module docstring."""
    out, pending = [], None
    for e in events:
        if pending is not None:
            if _is_restart(e):
                if e["start"] > pending:
                    out.append((pending, e["start"]))
                pending = None
                continue
            if e["type"] in OPEN_PLAY and e["start"] > pending + FPS:
                pending = None          # play went on: advantage, or a missed restart event
        if _is_trigger(e):
            pending = e["end"] if e["type"] in ("BALL OUT", "SHOT") else e["start"]
    return out


def labels(game, convention="strict"):
    """Per-frame 'H'/'A'/'D'/'X' truth: X outside the graded blocks."""
    n = len(game["ball"])
    truth = np.full(n, "X", dtype="<U1")
    events = game["events"]
    for p, first, last in blocks_of(game):
        ev = [e for e in events if e["period"] == p and first <= e["start"] <= last]
        owner = np.full(last - first + 1, -1, dtype=np.int8)
        current, streak_team, streak, streak_start = None, None, 0, first
        for e in ev:
            if e["type"] not in CONTROLLED:
                continue
            t = e["team"]
            if t == streak_team:
                streak += 1
            else:
                streak, streak_team, streak_start = 1, t, e["start"]
            if current is None or t == current:
                current = t
                owner[e["start"] - first:] = current
            elif convention == "loose" or e["type"] in STRONG_CONTROL:
                current = t
                owner[e["start"] - first:] = current
            elif streak >= 2:
                current = t              # control confirmed: the possession began at the first touch
                owner[streak_start - first:] = current
        if (owner < 0).any():
            first_known = owner[owner >= 0][0] if (owner >= 0).any() else HOME
            owner[owner < 0] = first_known
        lab = np.where(owner == HOME, "H", "A")
        for a, b in dead_intervals(ev):
            a, b = max(a, first), min(b, last + 1)
            if a < b:
                lab[a - first:b - first] = "D"
        truth[first:last + 1] = lab
    return truth
