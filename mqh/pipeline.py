"""End to end: tracking -> frame,label array for the whole video.

Owner: stream 5 (integration). Other streams plug in through two optional hooks, found
automatically when their files exist (see docs/INTERFACES.md):

    mqh/restarts.py  restart_dead_mask(n_frames, game)                      -> bool  (n_frames,)
    mqh/restarts.py  restart_evidence(n_frames, game)   (extra, optional)   -> float (n_frames,)
    mqh/gaps.py      gap_evidence(ball, players, team_of_slot, pitch, fps)  -> float (n_frames,)
    mqh/touches.py   touch_events(ball, players, team_of_slot, pitch, fps)  -> (frames, teams)

A hook that is absent is simply skipped. A hook that exists but returns the wrong shape or type
fails loudly with its name, so one stream cannot silently break another.
"""

import importlib
from dataclasses import dataclass, field

import numpy as np

from games import GAME, GAMES, half_video_blocks
from mqh.atd import load_atd
from mqh.dead import ball_speed_ms, dead_mask
from mqh.decode import team_evidence
from mqh.decoder import DecoderConfig, decode_half
from mqh.refine import snap_switches

# FAQ.md: the second half is graded from 110808 but kicks off at 113742; the gap is all dead ball.
SECOND_HALF_KICKOFF = 113742

# name: (module, function, required). A module that exists but lacks a *required* function is an
# error; a missing optional function is simply skipped.
HOOKS = {
    "restarts": ("mqh.restarts", "restart_dead_mask", True),
    "restart_team": ("mqh.restarts", "restart_evidence", False),
    "gaps": ("mqh.gaps", "gap_evidence", True),
    "touches": ("mqh.touches", "touch_events", True),
}


@dataclass(frozen=True)
class Config:
    """Settings of streams 2 and 3 (tracking evidence, tracking dead-ball) plus the decoder.

    Each stream keeps its own constants; stream 5 owns the decoder block. Values are reasoned, not
    tuned on this match: see docs/DECISIONS.md.
    """
    # evidence (stream 3)
    margin_scale_m: float = 2.0      # nearest-player distance gap that saturates the vote
    proximity_scale_m: float = 6.0   # how fast the vote fades when nobody is near the ball
    # dead ball from tracking (stream 2)
    use_pre_kickoff: bool = True
    use_out_of_bounds: bool = True
    oob_margin: float = 0.002        # normalised pitch units
    oob_close_frames: int = 50
    oob_tail_frames: int = 25
    use_stationary: bool = True
    still_speed_ms: float = 0.5
    still_min_frames: int = 75       # 3 s
    still_close_frames: int = 10
    # decoder (stream 5)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


def discover_hooks():
    """{name: function} for every hook module that exists. Absent modules are skipped."""
    found = {}
    for key, (module, attr, required) in HOOKS.items():
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name == module:
                continue
            raise
        if not hasattr(mod, attr):
            if required:
                raise AttributeError("{} exists but has no {}(); see docs/INTERFACES.md".format(
                    module, attr))
            continue
        found[key] = getattr(mod, attr)
    return found


def _checked(name, array, n_frames, kind):
    """Validate what a hook returned, so a bad contract names the culprit."""
    array = np.asarray(array)
    if array.shape != (n_frames,):
        raise ValueError("{}() must return shape ({},), got {}".format(
            name, n_frames, array.shape))
    if kind == "bool":
        if array.dtype != bool:
            raise ValueError("{}() must return a bool array, got {}".format(name, array.dtype))
    else:
        if not np.issubdtype(array.dtype, np.floating) and not np.issubdtype(array.dtype, np.integer):
            raise ValueError("{}() must return numbers, got {}".format(name, array.dtype))
        array = array.astype(float)
        if not np.isfinite(array).all():
            raise ValueError("{}() returned NaN or inf; use 0 where it has no opinion".format(name))
    return array


def _prepare_all(raw_path, meta_path, cfg, game, n_frames, hooks):
    """(evidence, dead, blocks, n_frames, touches) with touches = (frames, teams) or None."""
    n_frames = n_frames or GAMES[game]["n_video_frames"]
    pitch = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    hooks = discover_hooks() if hooks is None else hooks

    ball, players, team_of_slot = load_atd(raw_path, meta_path, n_frames)
    speed = ball_speed_ms(ball, pitch, fps)
    evidence = team_evidence(ball, players, team_of_slot, pitch, cfg)

    blocks = half_video_blocks(game)
    second_first = [b for b in blocks if b[0] == 2][0][1]
    dead = dead_mask(ball, speed, cfg, n_frames, pre_kickoff=(second_first, SECOND_HALF_KICKOFF - 1))

    if "restarts" in hooks:
        dead = dead | _checked("restart_dead_mask", hooks["restarts"](n_frames, game), n_frames, "bool")
    if "gaps" in hooks:
        gap = _checked("gap_evidence",
                       hooks["gaps"](ball, players, team_of_slot, pitch, fps), n_frames, "float")
        missing = np.isnan(ball[:, 0])
        evidence = np.where(missing, np.clip(gap, -1.0, 1.0), evidence)
    if "restart_team" in hooks:
        push = _checked("restart_evidence", hooks["restart_team"](n_frames, game), n_frames, "float")
        if np.abs(push).max() > 1.0:
            raise ValueError("restart_evidence() must stay within [-1, 1]")
        evidence = np.clip(evidence + push, -1.0, 1.0)
    touches = None
    if "touches" in hooks:
        touches = _checked_touches(hooks["touches"](ball, players, team_of_slot, pitch, fps), n_frames)
    return evidence, dead, blocks, n_frames, touches


def _checked_touches(result, n_frames):
    """Validate touch_events(): (frames, teams) of equal length, frames in range, teams 0 or 1."""
    try:
        frames, teams = result
        frames, teams = np.asarray(frames), np.asarray(teams)
    except (TypeError, ValueError):
        raise ValueError("touch_events() must return a pair (frames, teams)")
    if frames.ndim != 1 or frames.shape != teams.shape:
        raise ValueError("touch_events() frames and teams must be 1-D arrays of equal length")
    if frames.size == 0:
        return frames.astype(np.int64), teams.astype(np.int64)
    if not (np.issubdtype(frames.dtype, np.integer) and np.issubdtype(teams.dtype, np.integer)):
        raise ValueError("touch_events() must return integer arrays")
    if frames.min() < 0 or frames.max() >= n_frames:
        raise ValueError("touch_events() frames must lie in [0, {})".format(n_frames))
    if not np.isin(teams, [0, 1]).all():
        raise ValueError("touch_events() teams must be 0 (home) or 1 (away)")
    return frames.astype(np.int64), teams.astype(np.int64)


def prepare(raw_path, meta_path, cfg=Config(), game=GAME, n_frames=None, hooks=None):
    """Everything before the decoder: (evidence, dead mask, half blocks, n_frames).

    hooks=None finds the optional stream modules automatically; hooks={} disables them; a dict
    injects functions (used by the tests).
    """
    return _prepare_all(raw_path, meta_path, cfg, game, n_frames, hooks)[:4]


def decode(evidence, dead, blocks, n_frames, dcfg=DecoderConfig(), touches=None):
    """Labels for the whole video: decoded inside the graded halves, X everywhere else.

    touches=(frames, teams) from stream 3 snaps each team change onto the new team's first touch.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        half = decode_half(evidence, dead, first, last, dcfg)
        if touches is not None:
            inside = (touches[0] >= first) & (touches[0] <= last)
            half = snap_switches(half, touches[0][inside] - first, touches[1][inside],
                                 dcfg.snap_window_frames)
        labels[first:last + 1] = half
    return labels


def run(raw_path, meta_path, cfg=Config(), game=GAME, n_frames=None, hooks=None):
    evidence, dead, blocks, n_frames, touches = _prepare_all(
        raw_path, meta_path, cfg, game, n_frames, hooks)
    return decode(evidence, dead, blocks, n_frames, cfg.decoder, touches)


def write_csv(path, labels):
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
