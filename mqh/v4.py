"""The submission pipeline, v4: learned probabilities, tags and broadcast as evidence, joint decoding.

    labels = predict(raw_path, meta_path)          # what make_submission.py runs

1. Hand-built signals (streams 3, 3b, 4): team score, touches, team shape in ball gaps.
2. Two models trained on 30 external matches only (models/, made by tools/train_models.py):
   P(home has the ball) (mqh/team_model.py) and P(ball dead) (mqh/dead_model.py).
3. Smart Tagging (stream 1 v3): each set-piece tag's kick is found in the tracking; the stoppage
   before it is made dead, the first second after it alive, and the team named by the tag gets a
   push on its first two seconds of possession.
4. Optional extras, each validated before it is switched on (docs/DECISIONS.md):
   broadcast cutaways (mqh/broadcast.py), the referee's whistle (mqh/whistle.py).
5. Joint H / A / D Viterbi (mqh/decoder3.py), second-half pre-kick-off forced dead.

Deterministic: models are fixed files, the decoder works in integers.
"""

import os
import pickle
from dataclasses import dataclass, field

import numpy as np

from games import GAME, GAMES, half_video_blocks
from mqh import dead_model, restarts, team_model
from mqh.atd import load_atd
from mqh.decoder3 import Decoder3Config, decode3
from mqh.pipeline import ROOT, SECOND_HALF_KICKOFF, Config, discover_hooks, prepare_arrays

MODELS = os.path.join(ROOT, "models")


@dataclass(frozen=True)
class V4Config:
    decoder: Decoder3Config = field(default_factory=lambda: Decoder3Config(team_cost=10.0, stop_cost=20.0,
                                                                           restart_cost=20.0, min_dead_frames=25))
    tag_dead: float = 0.95        # p_dead on a tagged stoppage, up to the kick
    tag_alive_s: float = 1.0      # after the kick: alive for this long
    tag_team_push: float = 0.25   # P(home) moved this far towards the tagged team after the kick
    tag_team_s: float = 2.0
    use_tags: bool = True         # on BAR-ATH the learned p_dead covers only half the tagged stoppages (DECISIONS.md)
    cutaway_fusion: tuple = ()    # (a, b, c, d) logit fusion weights, () = off
    model_dir: str = MODELS


def load_models(model_dir=MODELS):
    with open(os.path.join(model_dir, "team_model.pkl"), "rb") as handle:
        tm = pickle.load(handle)
    with open(os.path.join(model_dir, "dead_model.pkl"), "rb") as handle:
        dm = pickle.load(handle)
    return tm, dm


def probabilities(ball, players, team_of_slot, pitch, fps, blocks, models, base_cfg=Config()):
    """(p_home, p_dead, touches) from tracking arrays: the part shared with the external benches."""
    hooks = {k: v for k, v in discover_hooks().items() if k in ("gaps", "touches")}
    evidence, _, _, n, touches = prepare_arrays(ball, players, team_of_slot, base_cfg, pitch, fps, blocks,
                                                hooks=hooks)
    tm, dm = models
    anti, sym = team_model.team_features(ball, players, team_of_slot, pitch, fps, evidence)
    p_home = (team_model.evidence(tm, anti, sym) + 1.0) / 2.0
    p_dead = dead_model.probability(dm, dead_model.features(ball, players, team_of_slot, pitch, fps))
    return p_home, p_dead, touches


def apply_tags(p_home, p_dead, windows, ball, pitch, fps, cfg=V4Config()):
    """Stream 1 v3 as evidence: (p_home, p_dead, stoppages) after the tagged restarts."""
    p_home, p_dead = p_home.copy(), p_dead.copy()
    n = len(p_dead)
    rcfg = restarts.RestartConfig()
    stops = restarts.stoppages(windows, n, rcfg, fps, track_dead=p_dead >= 0.5,
                               ball_speed=restarts.kick_speed(ball, pitch, fps))
    alive = int(cfg.tag_alive_s * fps)
    push = int(cfg.tag_team_s * fps)
    for _, a, kick, team in stops:
        p_dead[a:kick] = np.maximum(p_dead[a:kick], cfg.tag_dead)
        p_dead[kick:kick + alive] = np.minimum(p_dead[kick:kick + alive], 1.0 - cfg.tag_dead)
        if team:
            sl = slice(kick, min(n, kick + push))
            p_home[sl] = np.clip(p_home[sl] + team * cfg.tag_team_push, 0.0, 1.0)
    return p_home, p_dead, stops


def fuse_cutaway(p_dead, cutaway, weights):
    from scipy.ndimage import uniform_filter1d
    a, b, c, d = weights
    p = np.clip(p_dead, 1e-3, 1 - 1e-3)
    z = a * np.log(p / (1 - p)) + b * cutaway + c * uniform_filter1d(cutaway.astype(float), 101) + d
    return 1.0 / (1.0 + np.exp(-z))


def predict(raw_path, meta_path, cfg=V4Config(), game=GAME, extras=None):
    """Labels for the whole video; `extras` may carry {"cutaway": bool array} (mqh/broadcast.py)."""
    n = GAMES[game]["n_video_frames"]
    pitch, fps = GAMES[game]["pitch"], GAMES[game]["fps"]
    ball, players, team_of_slot = load_atd(raw_path, meta_path, n)
    blocks = half_video_blocks(game)
    second_first = [b for b in blocks if b[0] == 2][0][1]
    p_home, p_dead, _ = probabilities(ball, players, team_of_slot, pitch, fps, blocks, load_models(cfg.model_dir))
    if cfg.use_tags:
        events, tag_fps = restarts._events(game)
        p_home, p_dead, _ = apply_tags(p_home, p_dead, restarts.restart_windows(events), ball, pitch, tag_fps, cfg)
    if cfg.cutaway_fusion and extras and "cutaway" in extras:
        p_dead = fuse_cutaway(p_dead, extras["cutaway"], cfg.cutaway_fusion)
    return decode3(p_home, p_dead, blocks, n, cfg.decoder, pre_kickoff=(second_first, SECOND_HALF_KICKOFF - 1))
