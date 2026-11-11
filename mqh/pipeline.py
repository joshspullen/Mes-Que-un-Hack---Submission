"""End to end: tracking -> frame,label array for the whole video."""

from dataclasses import dataclass

import numpy as np

from games import GAME, GAMES, half_video_blocks
from mqh.atd import load_atd
from mqh.dead import ball_speed_ms, dead_mask
from mqh.decode import decode_half, team_evidence

# FAQ.md: the second half is graded from 110808 but kicks off at 113742; the gap is all dead ball.
SECOND_HALF_KICKOFF = 113742


@dataclass(frozen=True)
class Config:
    """Every knob in one place. v0 values are reasoned, not tuned on this match."""
    # evidence
    margin_scale_m: float = 2.0      # nearest-player distance gap that saturates the vote
    proximity_scale_m: float = 6.0   # how fast the vote fades when nobody is near the ball
    # decoder: ~ frames of consistent contrary evidence needed to switch team (25 fps)
    switch_penalty: float = 30.0
    # dead ball
    use_pre_kickoff: bool = True
    use_out_of_bounds: bool = True
    oob_margin: float = 0.002        # normalised pitch units
    oob_close_frames: int = 50
    oob_tail_frames: int = 25
    use_stationary: bool = True
    still_speed_ms: float = 0.5
    still_min_frames: int = 75       # 3 s
    still_close_frames: int = 10


def run(raw_path, meta_path, cfg=Config(), game=GAME, n_frames=None):
    n_frames = n_frames or GAMES[game]["n_video_frames"]
    pitch = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]

    ball, players, team_of_slot = load_atd(raw_path, meta_path, n_frames)
    speed = ball_speed_ms(ball, pitch, fps)
    evidence = team_evidence(ball, players, team_of_slot, pitch, cfg)

    blocks = half_video_blocks(game)
    second_first = [b for b in blocks if b[0] == 2][0][1]
    pre = (second_first, SECOND_HALF_KICKOFF - 1)
    dead = dead_mask(ball, speed, cfg, n_frames, pre_kickoff=pre)

    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        labels[first:last + 1] = decode_half(evidence, dead, first, last, cfg)
    return labels


def write_csv(path, labels):
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
