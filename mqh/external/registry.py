"""All external matches behind one interface, and a disk cache of everything derived from them.

Three providers, 30 matches, three ways of tracking and three ways of labelling:
  metrica      3 games   Metrica Sports sample data   optical, ball hidden at stoppages, labels from events
  skillcorner  20 games  SkillCorner open data        broadcast, players extrapolated, labels from phases of play
  idsse        7 games   DFL / IDSSE (TRACAB)          optical, ball always tracked, DFL ball status + possession

A choice that only helps one provider or one labelling is not a choice worth making for a match
nobody can see the labels of; every bench here groups by match and reports per provider.

    load(provider, match)        -> dict(name, provider, ball, players, team_of_slot, blocks, truths)
    cases(profile, seed, ...)    -> degraded copies, cached on disk (.cache/external/)
"""

import os
import pickle

import numpy as np

from mqh.external import idsse, metrica, skillcorner
from mqh.external.degrade import PROFILES, degrade

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "external")
CACHE = os.path.join(ROOT, ".cache", "external")

PROVIDERS = {
    "metrica": dict(module=metrica, folder="metrica-sample-data", conventions=("strict", "loose")),
    "skillcorner": dict(module=skillcorner, folder="skillcorner-opendata", conventions=("phases",)),
    "idsse": dict(module=idsse, folder="idsse", conventions=("strict", "raw")),
}


def matches(provider):
    p = PROVIDERS[provider]
    root = os.path.join(DATA, p["folder"])
    if not os.path.isdir(root):
        return []
    if provider == "metrica":
        return [1, 2, 3] if os.path.isdir(os.path.join(root, "data")) else []
    return p["module"].match_ids(root)


def all_matches(providers=tuple(PROVIDERS)):
    return [(p, m) for p in providers for m in matches(p)]


def load(provider, match):
    """Clean (undegraded) match with labels for every convention of its provider. Cached."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "clean-{}-{}.pkl".format(provider, match))
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    p = PROVIDERS[provider]
    mod = p["module"]
    g = mod.load_game(match, os.path.join(DATA, p["folder"]))
    truths = {}
    for conv in p["conventions"]:
        truths[conv] = mod.labels(g, conv)
    out = {"name": g["name"], "provider": provider, "ball": g["ball"].astype(np.float32),
           "players": g["players"].astype(np.float32), "team_of_slot": g["team_of_slot"],
           "blocks": mod.blocks_of(g), "truths": truths}
    with open(path, "wb") as handle:
        pickle.dump(out, handle, protocol=4)
    return out


def main_truth(game):
    """The convention used for training: the first of the provider's list."""
    return game["truths"][PROVIDERS[game["provider"]]["conventions"][0]]


def degraded(game, profile, seed):
    """(ball, players) degraded like our tracking under `profile` (mqh/external/degrade.py)."""
    live = np.isin(main_truth(game), ["H", "A"])
    ball, players = degrade(game["ball"].astype(float), game["players"].astype(float), live,
                            seed=seed, cfg=PROFILES[profile])
    return ball, players
