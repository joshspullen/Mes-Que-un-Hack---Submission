"""IDSSE / DFL open data (7 Bundesliga matches, CC-BY 4.0) -> our arrays + H/A/D labels.

Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data 12, 195 (doi:10.1038/s41597-025-04505-y). Data owner: Deutsche
Fussball Liga (DFL). Mirror used to download: huggingface.co/datasets/pysport/idsse-data.
Files are kept in data/external/idsse/ (git-ignored); positions are gzipped there.

Official TRACAB optical tracking at 25 fps, every player and the ball, plus the DFL's own per-frame
ball status (alive / dead) and team in possession on the ball's frames. A third provider next to
Metrica and SkillCorner, and the only one whose dead-ball and possession flags come with the
tracking itself.

Labels, by code from those flags:
  D      BallStatus = 0
  H / A  BallPossession = 1 / 2 (DFL's team in possession)
  strict reading: a possession run shorter than 1 s inside live play is given to the team before
         it (the DFL flag follows every touch; the challenge counts controlled possession only)
"""

import gzip
import os
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
HOME, AWAY = 0, 1
MIN_SPELL_S = 1.0


def match_ids(root):
    return sorted(f.split("DFL-MAT-")[1].split(".")[0] for f in os.listdir(root)
                  if f.startswith("DFL_04_03_positions") and "DFL-MAT-" in f)


def _find(root, kind, mid):
    for f in os.listdir(root):
        if f.startswith(kind) and "DFL-MAT-{}".format(mid) in f:
            return os.path.join(root, f)
    raise IOError("missing {} for {}".format(kind, mid))


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def load_game(mid, root):
    info = ET.parse(_find(root, "DFL_02_01_matchinformation", mid)).getroot()
    general = info.find(".//General")
    home_id, away_id = general.get("HomeTeamId"), general.get("GuestTeamId")
    env = info.find(".//Environment")
    length = float(env.get("PitchX", 105.0))
    width = float(env.get("PitchY", 68.0))
    person_team = {}
    for team in info.iter("Team"):
        tid = team.get("TeamId")
        for p in team.iter("Player"):
            person_team[p.get("PersonId")] = HOME if tid == home_id else AWAY

    frames = {}       # (section, person) -> (N array, X array, Y array)
    ball_flags = {}   # section -> (N, possession, status)
    with _open(_find(root, "DFL_04_03_positions", mid)) as handle:
        for _, el in ET.iterparse(handle, events=("end",)):
            if el.tag != "FrameSet":
                continue
            section, person, team = el.get("GameSection"), el.get("PersonId"), el.get("TeamId")
            kids = list(el)
            N = np.fromiter((int(k.get("N")) for k in kids), dtype=np.int64, count=len(kids))
            X = np.fromiter((float(k.get("X")) for k in kids), dtype=float, count=len(kids))
            Y = np.fromiter((float(k.get("Y")) for k in kids), dtype=float, count=len(kids))
            if team is not None and team.upper() == "BALL":
                pos = np.fromiter((int(k.get("BallPossession") or 0) for k in kids), dtype=np.int8, count=len(kids))
                st = np.fromiter((int(k.get("BallStatus") or 0) for k in kids), dtype=np.int8, count=len(kids))
                ball_flags[section] = (N, pos, st)
                frames[(section, "BALL")] = (N, X, Y)
            elif person in person_team:
                frames[(section, person)] = (N, X, Y)
            el.clear()

    sections = [s for s in ("firstHalf", "secondHalf") if s in ball_flags]
    persons = sorted({p for (_, p) in frames if p != "BALL"})
    slot = {p: i for i, p in enumerate(persons)}
    starts, n = {}, 0
    for s in sections:
        N = ball_flags[s][0]
        starts[s] = (n, int(N.min()))
        n += int(N.max() - N.min()) + 1 + 2 * FPS          # a 2 s gap between the halves
    ball = np.full((n, 2), np.nan)
    players = np.full((n, len(persons), 2), np.nan)
    period = np.zeros(n, dtype=np.int8)
    possession = np.zeros(n, dtype=np.int8)
    status = np.zeros(n, dtype=np.int8)

    def put(arr, s, N, values):
        off, n0 = starts[s]
        idx = off + (N - n0)
        ok = (idx >= 0) & (idx < n)
        arr[idx[ok]] = values[ok]

    for (s, p), (N, X, Y) in frames.items():
        if s not in starts:
            continue
        xy = np.column_stack([X / length + 0.5, 0.5 - Y / width])   # centred metres -> [0, 1]
        if p == "BALL":
            put(ball, s, N, xy)
        else:
            put(players[:, slot[p]], s, N, xy)
    for k, s in enumerate(sections):
        N, pos, st = ball_flags[s]
        put(period, s, N, np.full(len(N), k + 1, dtype=np.int8))
        put(possession, s, N, pos)
        put(status, s, N, st)
    # the dead ball is still tracked by TRACAB; a real tracker loses it out of the pitch -- the
    # degradation profiles take care of that, nothing is removed here
    team_of_slot = np.array([person_team[p] for p in persons], dtype=np.int8)
    return {"name": "idsse{}".format(mid), "ball": ball, "players": players,
            "team_of_slot": team_of_slot, "period": period, "possession": possession,
            "status": status}


def blocks_of(game):
    """[(period, first, last)]: from the first alive frame of each half to its last frame."""
    out = []
    for p in (1, 2):
        idx = np.flatnonzero(game["period"] == p)
        alive = idx[game["status"][idx] == 1]
        if alive.size:
            out.append((p, int(alive[0]), int(idx[-1])))
    return out


def labels(game, convention="strict"):
    n = len(game["ball"])
    truth = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks_of(game):
        st = game["status"][first:last + 1]
        pos = game["possession"][first:last + 1].astype(int)
        lab = np.where(pos == 1, "H", np.where(pos == 2, "A", "?")).astype("<U1")
        # unknown team while alive -> previous team
        cur = "H"
        for i in range(len(lab)):
            if lab[i] == "?":
                lab[i] = cur
            else:
                cur = lab[i]
        if convention == "strict":
            short = int(MIN_SPELL_S * FPS)
            live = st == 1
            i = 0
            prev = None
            while i < len(lab):
                j = i
                while j < len(lab) and lab[j] == lab[i] and live[j] == live[i]:
                    j += 1
                if live[i] and prev is not None and lab[i] != prev and j - i < short and \
                        j < len(lab) and lab[j] == prev and live[j]:
                    lab[i:j] = prev
                elif live[i]:
                    prev = lab[i]
                i = j
        lab[st != 1] = "D"
        truth[first:last + 1] = lab
    return truth
