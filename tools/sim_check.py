"""Label-free statistics of BAR-ATH's tracking next to the degraded external games.

    python tools/sim_check.py            # needs the match data locally; prints numbers only

The external bench (tools/external_bench.py) is only as good as the degradation that makes
Metrica's games look like our ATD (mqh/external/degrade.py). This compares quantities that need no
labels: how often the ball is missing and for how long, how many players are tracked, how jittery
positions are, how much dead ball stream 2 finds, how many possessions the pipeline reports. If a
number differs a lot, change DegradeConfig (with the reason in docs/DECISIONS.md) and rerun the
bench / fits. Run it locally: it prints aggregates only, never frame-level data.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
from external_bench import DEFAULT_DIR, build_inputs, fetch  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from mqh.atd import load_atd  # noqa: E402
from mqh.dead import tracking_dead_mask  # noqa: E402
from mqh.external import metrica  # noqa: E402
from mqh.pipeline import Config, decode, prepare_arrays  # noqa: E402


def stats(name, ball, players, team_of_slot, blocks, pitch, fps):
    graded = np.zeros(len(ball), dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    miss = np.isnan(ball[:, 0])
    pad = np.concatenate(([0], (miss & graded).astype(np.int8), [0]))
    d = np.diff(pad)
    lengths = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) / fps
    tracked = (~np.isnan(players[:, :, 0])).sum(axis=1)[graded].mean()
    p = players * np.asarray(pitch)
    step = np.hypot(*np.moveaxis(p[2:] - 2 * p[1:-1] + p[:-2], -1, 0))   # second difference: jitter
    jitter = np.nanmedian(step[graded[1:-1]])
    dead = tracking_dead_mask(ball, players, pitch, fps, len(ball))
    ev, dm, bl, n, touches = prepare_arrays(ball, players, team_of_slot, Config(), pitch, fps, blocks, hooks={})
    lab = decode(ev, dm, bl, n, Config().decoder)
    alive = np.isin(lab, ["H", "A"])
    spells = int(alive[0]) + int(((lab[1:] != lab[:-1]) & alive[1:]).sum())
    minutes = graded.sum() / fps / 60
    print("{:<16} ball missing {:5.1%}  gaps/min {:4.1f}  gap median {:4.1f}s p90 {:5.1f}s  players {:4.1f}  "
          "jitter {:.3f} m  dead (stream 2) {:5.1%}  possessions/min {:4.2f}".format(
              name, miss[graded].mean(), len(lengths) / minutes, np.median(lengths) if len(lengths) else 0,
              np.percentile(lengths, 90) if len(lengths) else 0, tracked, jitter, dead[graded].mean(),
              spells / minutes))


def main():
    try:
        meta, raw = require(["atd_meta", "atd_raw"])
        n = GAMES[GAME]["n_video_frames"]
        ball, players, team = load_atd(raw, meta, n)
        stats("BAR-ATH", ball, players, team, half_video_blocks(GAME), GAMES[GAME]["pitch"], GAMES[GAME]["fps"])
        # Does our tracking hide the ball at stoppages, as Metrica's sample data does (90-99% of
        # dead frames)? If the ball is missing far more inside tagged set-piece windows than
        # elsewhere, "no ball" is a strong dead-ball signal here too, and the no_ball / gap_log
        # features of stream 2 (mqh/dead.py, left at 0) could be switched back on.
        try:
            from mqh.restarts import restart_events
            from starter.load_smart_tagging import load as load_tags
            inside = np.zeros(n, dtype=bool)
            for _, a, b in restart_events(load_tags(GAME, None, GAMES[GAME]["fps"])):
                inside[max(a, 0):min(b, n)] = True
            graded = np.zeros(n, dtype=bool)
            for _, a, b in half_video_blocks(GAME):
                graded[a:b + 1] = True
            miss = np.isnan(ball[:, 0])
            print("BAR-ATH ball missing inside SET PIECES windows {:.1%}, elsewhere {:.1%}".format(
                miss[graded & inside].mean(), miss[graded & ~inside].mean()))
        except Exception as error:   # the check is optional
            print("set-piece check skipped:", error)
    except IOError as error:
        print("BAR-ATH data not found, external games only:", str(error).splitlines()[0])
    games = [metrica.load_game(k, fetch(DEFAULT_DIR)) for k in (1, 2, 3)]
    for name, ball, players, team, blocks, _ in build_inputs(games, [0]):
        stats(name, ball, players, team, blocks, metrica.PITCH, metrica.FPS)


if __name__ == "__main__":
    main()
