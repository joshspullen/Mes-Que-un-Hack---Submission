"""Dead ball from Smart Tagging: a yes/no mask, one value per frame, marking the stoppage before
each throw-in, goal kick, corner, free kick and kick-off.

The tagging is read with the organisers' loader (starter/load_smart_tagging.py, MIT, copied in
unmodified -- see THIRD_PARTY_NOTICES.md). It is automatic, not ground truth: a tag is evidence
that a restart happened near that time, so this mask is one vote among the dead-ball signals, not
a label on its own.

How a tag becomes dead frames:
  1. restart frame = tag start + anchor_frac * tag length. The organisers' starter says the tag is
     a ~5 s window *around* the restart, so the default anchor is the middle of the tag.
  2. the stoppage is the `pre_dead_s[type]` seconds before the restart frame, up to (not including)
     the restart itself, when the ball is live again.

Both the anchor and the per-type durations are UNTUNED GUESSES from football knowledge. Step 1
(tools/inspect_set_pieces.py) is what checks the anchor; the durations should come from the
tracking (when the ball actually went out), not from these constants, once that detector exists.
Nothing here is set from the public ground truth.
"""

from dataclasses import dataclass, field

import numpy as np

from starter.load_smart_tagging import load as load_smart_tagging

SET_PIECES = "set pieces"

# Typical time the ball is dead before each kind of restart, in seconds. Guesses, see above.
DEFAULT_PRE_DEAD_S = {
    "throw-in": 8.0,
    "corner kick": 15.0,
    "goal kick": 15.0,
    "free kick": 20.0,
    "kick off (start)": 0.0,       # half starts: graded from the kick-off (1st) or covered by
                                   # the pre-kickoff rule (2nd), nothing to add
    "kick off (after goal)": 45.0,  # celebration + walk back
}


@dataclass(frozen=True)
class RestartConfig:
    anchor_frac: float = 0.5
    pre_dead_s: dict = field(default_factory=lambda: dict(DEFAULT_PRE_DEAD_S))


def _lower(value):
    return (value or "").strip().lower()


def restart_events(events):
    """SET PIECES instances with a known restart Type, as (restart_type, start_frame, end_frame).

    Matching is case-insensitive (codes are upper case, values title case in the file).
    """
    known = set(DEFAULT_PRE_DEAD_S)
    out = []
    for e in events:
        kind = _lower(e["tags"].get("Type"))
        if _lower(e["code"]) == SET_PIECES and kind in known:
            out.append((kind, e["start_frame"], e["end_frame"]))
    return out


def restart_frames(events, cfg=RestartConfig()):
    """[(restart_type, restart_frame)] -- the estimated instant each restart is taken."""
    return [(kind, int(round(a + cfg.anchor_frac * (b - a))))
            for kind, a, b in restart_events(events)]


def dead_mask_from_tags(events, n_frames, cfg=RestartConfig(), fps=25):
    """Boolean per-frame mask: True on the estimated stoppage before each tagged restart."""
    dead = np.zeros(n_frames, dtype=bool)
    for kind, restart in restart_frames(events, cfg):
        pre = int(round(cfg.pre_dead_s.get(kind, 0.0) * fps))
        a, b = max(restart - pre, 0), min(restart, n_frames)
        if a < b:
            dead[a:b] = True
    return dead


def load_mask(game=None, data_dir=None, n_frames=None, cfg=RestartConfig()):
    """Convenience: read the game's tagging file and return the mask over the whole video."""
    from games import GAME, GAMES
    game = game or GAME
    fps = GAMES[game]["fps"]
    events = load_smart_tagging(game, data_dir, fps)
    return dead_mask_from_tags(events, n_frames or GAMES[game]["n_video_frames"], cfg, fps)


def restart_dead_mask(n_frames, game):
    """Stream 1 hook (docs/INTERFACES.md): bool (n_frames,), True on the estimated stoppages.

    Without this name the pipeline's hook discovery refused to run (`mqh.restarts exists but has no
    restart_dead_mask()`). Before trusting the mask, run tools/inspect_set_pieces.py on the real
    tagging: if the SET PIECES tags are long fixed-length clips rather than ~5 s windows around the
    restart, the default anchor (the middle of the tag) is far from the real restart.
    """
    return load_mask(game=game, n_frames=n_frames)
