"""Read the raw ATD (FIFA/EPTS text export) into dense numpy arrays on the video timeline.

Raw line: "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,speed" with NaN for a missing
observation. frame_id counts from 1, submissions count from 0, so video_frame = frame_id - 1.
Coordinates are normalised 0-1; multiply by the pitch (metres) before measuring distances.

The team-of-slot logic follows starter/baseline.py (first <Team> in the metadata is home).
"""

import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """int8 array over track slots: 0 = home, 1 = away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def _parse_groups(text, n_slots):
    """'x,y,s;x,y,s;...' -> (n_slots, 3) floats. Tolerates a bare 'NaN' group."""
    flat = text.replace(";", ",")
    try:
        arr = np.fromstring(flat, sep=",")
        if arr.size == 3 * n_slots:
            return arr.reshape(n_slots, 3)
    except ValueError:
        pass
    out = np.full((n_slots, 3), np.nan)
    for i, group in enumerate(text.split(";")[:n_slots]):
        vals = group.split(",")
        if len(vals) == 3:
            try:
                out[i] = [float(v) for v in vals]
            except ValueError:
                pass
    return out


def load_atd(raw_path, meta_path, n_frames):
    """Return (ball, players, team_of_slot).

    ball:    (n_frames, 2) normalised x, y; NaN where the ball is not tracked
    players: (n_frames, n_slots, 2) normalised x, y; NaN where a slot is empty
    """
    team_of_slot = team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            players[frame] = _parse_groups(parts[1], n_slots)[:, :2]
            b = parts[2].strip().split(",")
            if len(b) >= 2:
                try:
                    ball[frame] = (float(b[0]), float(b[1]))
                except ValueError:
                    pass
    return ball, players, team_of_slot
