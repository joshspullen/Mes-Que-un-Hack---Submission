"""Dead ball from Smart Tagging: a yes/no mask, one value per frame, marking the stoppage before
each throw-in, goal kick, corner, free kick and kick-off. Optionally, a push towards the team that
takes the restart.

The tagging is read with the organisers' loader (starter/load_smart_tagging.py, MIT, copied in
unmodified -- see THIRD_PARTY_NOTICES.md). It is automatic, not ground truth: a tag is evidence
that a restart happened near that time, so this mask is one vote among the dead-ball signals, not
a label on its own.

How a tag becomes dead frames (v2, `use_tracking=True`, the default):
  1. The organisers' starter says a SET PIECES tag is a ~5 s window *around* the restart. So the
     restart lies inside [tag start, tag end).
  2. The tracking dead-ball detector (stream 2, mqh/dead.py) usually sees the stoppage too, but in
     pieces, and it does not know where the stoppage ends. The tag tells it. We take the tracking
     dead runs (holes under `bridge_s` filled) that reach the tag window, at most `search_before_s`
     before the restart, and mark them dead from their first frame to the restart, the restart
     being where the tracking says play resumed, kept within `tolerance_s` of the tag window
     (real tags may not be centred on the restart; the tolerance makes a 3 s offset harmless).
  3. If the tracking sees no stoppage near the tag, fall back to v1: `pre_dead_s[type]` seconds
     before the tag's middle (`anchor_frac`).
v1 (`use_tracking=False`) is step 3 alone.

Restart team (`restart_evidence`): every set piece is a restart by one team, and the tag carries
the team except for throw-ins. For `push_s` seconds after the restart the evidence is pushed by
`push` towards that team. Team tag text is matched to home/away by name (HOME_NAMES / AWAY_NAMES);
anything else gives no push.

Every constant was chosen on EXTERNAL matches with simulated tags (tools/bench_restarts.py, Metrica
sample games, tags = ~5 s windows around the real restart, 10% missed). Nothing here is set from
this match's ground truth or video. Reasons and numbers: docs/DECISIONS.md, "Stream 1".
"""

from dataclasses import dataclass, field

import numpy as np

from starter.load_smart_tagging import load as load_smart_tagging

SET_PIECES = "set pieces"

# v1 fallback: typical time the ball is dead before each kind of restart, in seconds. Guesses from
# football knowledge; only used when the tracking sees no stoppage near a tag. 0 = skip the type.
DEFAULT_PRE_DEAD_S = {
    "throw-in": 8.0,
    "corner kick": 15.0,
    "goal kick": 15.0,
    "free kick": 20.0,
    "penalty": 30.0,
    "kick off (start)": 0.0,       # half starts: graded from the kick-off (1st) or covered by
                                   # the pre-kickoff rule (2nd), nothing to add
    "kick off (after goal)": 45.0,  # celebration + walk back
}

# Team tag text -> side. BAR-ATH: Barcelona at home. Lower-case substrings.
HOME_NAMES = ("barcelona", "barça", "barca")
AWAY_NAMES = ("athletic", "bilbao")


@dataclass(frozen=True)
class RestartConfig:
    anchor_frac: float = 0.5
    pre_dead_s: dict = field(default_factory=lambda: dict(DEFAULT_PRE_DEAD_S))
    use_tracking: bool = True
    search_before_s: float = 60.0   # longest stoppage we let the tracking extend back to
    bridge_s: float = 3.0           # holes in a tracking stoppage shorter than this are filled
    tolerance_s: float = 2.0        # tracking stoppages may reach this far outside the tag window
    push: float = 0.5               # restart-team push on the evidence, 0 = off
    push_s: float = 2.0             # seconds after the restart that the push lasts


def _lower(value):
    return (value or "").strip().lower()


def team_sign(text):
    """+1 home, -1 away, 0 unknown, from a Smart Tagging Team value."""
    t = _lower(text)
    if any(n in t for n in HOME_NAMES):
        return 1
    if any(n in t for n in AWAY_NAMES):
        return -1
    return 0


def restart_windows(events):
    """SET PIECES tags with a known restart Type, as (restart_type, start_frame, end_frame, team).

    team is +1 / -1 / 0 (team_sign of the Team tag; throw-ins carry none). Matching is
    case-insensitive (codes are upper case, values title case in the file).
    """
    known = set(DEFAULT_PRE_DEAD_S)
    out = []
    for e in events:
        kind = _lower(e["tags"].get("Type"))
        if _lower(e["code"]) == SET_PIECES and kind in known:
            out.append((kind, e["start_frame"], e["end_frame"], team_sign(e["tags"].get("Team"))))
    return out


def restart_events(events):
    """(restart_type, start_frame, end_frame) -- restart_windows without the team."""
    return [w[:3] for w in restart_windows(events)]


def restart_frames(events, cfg=RestartConfig()):
    """[(restart_type, restart_frame)] -- the tag's anchor point (v1 estimate of the restart)."""
    return [(kind, int(round(a + cfg.anchor_frac * (b - a))))
            for kind, a, b in restart_events(events)]


def _runs(mask):
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(padded)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _bridge(mask, max_gap):
    out = mask.copy()
    s, e = _runs(mask)
    for i in range(len(s) - 1):
        if s[i + 1] - e[i] <= max_gap:
            out[e[i]:s[i + 1]] = True
    return out


def stoppages(windows, n_frames, cfg=RestartConfig(), fps=25, track_dead=None):
    """[(kind, dead_start, restart, team)] for every usable tag: [dead_start, restart) is dead."""
    closed = None
    if cfg.use_tracking and track_dead is not None:
        closed = _bridge(np.asarray(track_dead, dtype=bool), int(round(cfg.bridge_s * fps)))
    bridge = int(round(cfg.bridge_s * fps))
    tol = int(round(cfg.tolerance_s * fps))
    back = int(round(cfg.search_before_s * fps))
    out = []
    for kind, a, b, team in windows:
        pre_s = cfg.pre_dead_s.get(kind, 0.0)
        if pre_s <= 0:
            continue
        a, b = max(a, 0), min(b, n_frames)
        if a >= b:
            continue
        anchor = int(round(a + cfg.anchor_frac * (b - a)))
        found = None
        if closed is not None:
            lo = max(anchor - back, 0)
            hi = min(b + tol, n_frames)
            s, e = _runs(closed[lo:hi])
            s, e = s + lo, e + lo
            near = (e > a - bridge - tol) & (s < b)   # reaches the (widened) tag window
            if near.any():
                start, end = s[near][0], e[near][-1]
                found = (int(start), int(min(max(end, a - tol), b + tol)))
        if found is None:
            pre = int(round(pre_s * fps))
            found = (max(anchor - pre, 0), anchor)
        if found[0] < found[1]:
            out.append((kind, found[0], found[1], team))
    return out


def dead_from_stoppages(stops, n_frames):
    dead = np.zeros(n_frames, dtype=bool)
    for _, a, b, _ in stops:
        dead[a:b] = True
    return dead


def push_from_stoppages(stops, n_frames, cfg=RestartConfig(), fps=25):
    """float (n_frames,): +-cfg.push for push_s seconds after each restart with a known team."""
    push = np.zeros(n_frames)
    if cfg.push <= 0:
        return push
    length = int(round(cfg.push_s * fps))
    for _, _, restart, team in stops:
        if team:
            push[restart:min(restart + length, n_frames)] = team * cfg.push
    return np.clip(push, -1.0, 1.0)


def dead_mask_from_tags(events, n_frames, cfg=RestartConfig(), fps=25, track_dead=None):
    """Boolean per-frame mask: True on the estimated stoppage before each tagged restart."""
    return dead_from_stoppages(stoppages(restart_windows(events), n_frames, cfg, fps, track_dead),
                               n_frames)


def _events(game, data_dir=None):
    from games import GAMES
    fps = GAMES[game]["fps"]
    return load_smart_tagging(game, data_dir, fps), fps


def load_mask(game=None, data_dir=None, n_frames=None, cfg=RestartConfig(), track_dead=None):
    """Convenience: read the game's tagging file and return the mask over the whole video."""
    from games import GAME, GAMES
    game = game or GAME
    events, fps = _events(game, data_dir)
    return dead_mask_from_tags(events, n_frames or GAMES[game]["n_video_frames"], cfg, fps,
                               track_dead)


def restart_dead_mask(n_frames, game, track_dead=None):
    """Stream 1 hook (docs/INTERFACES.md): bool (n_frames,), True on the estimated stoppages.

    track_dead is the tracking dead-ball mask (stream 2); the pipeline passes it so the tags can be
    anchored on it. Without it this is the v1 fixed-duration mask.
    """
    return load_mask(game=game, n_frames=n_frames, track_dead=track_dead)


def restart_evidence(n_frames, game, track_dead=None, cfg=RestartConfig()):
    """Stream 1b hook: float (n_frames,) in [-1, 1], a push towards the team taking each restart.

    All zeros if RestartConfig.push is 0 or no Team tag matches HOME_NAMES / AWAY_NAMES.
    """
    if cfg.push <= 0:
        return np.zeros(n_frames)
    events, fps = _events(game)
    stops = stoppages(restart_windows(events), n_frames, cfg, fps, track_dead)
    return push_from_stoppages(stops, n_frames, cfg, fps)

