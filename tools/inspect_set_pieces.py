"""Step 1 of the dead-ball work: where does a Smart Tagging restart sit relative to the real restart?

For every SET PIECES and GOALS instance, plots the tracking from 15 s before the tag to 10 s after
it, so you can see by eye whether the tag starts at the restart, before it or after it:

    * ball speed (m/s) and where the ball is untracked
    * median player speed (m/s) -- players slow down at a whistle and speed up at the restart
    * ball distance to the nearest pitch line (m, negative = outside)

Also prints one line per tag with its video time (mm:ss) so you can scrub the video to it.

    pip install matplotlib
    python tools/inspect_set_pieces.py                 # all restarts
    python tools/inspect_set_pieces.py --type throw-in # one kind only
    python tools/inspect_set_pieces.py --max 10

Figures go to inspect/ which is gitignored: they are frame-level views of the match data and must
not be committed or shared outside the team (competition terms). Nothing here reads any ground
truth, and nothing here should become a frame-specific rule -- it is for understanding the tags.
"""

import argparse
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402
from mqh.atd import load_atd  # noqa: E402
from mqh.dead import ball_speed_ms  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

BEFORE_S, AFTER_S = 15, 10


def median_player_speed(players, pitch, fps):
    """Per-frame median over slots of each slot's speed (m/s). Median, not mean, so an identity
    switch (a slot jumping across the pitch) does not spike it."""
    xy = players * np.array(pitch)
    step = np.hypot(np.diff(xy[..., 0], axis=0), np.diff(xy[..., 1], axis=0)) * fps
    step[step > 12] = np.nan  # faster than any sprint: a track jump, not a run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no players tracked
        med = np.nanmedian(step, axis=1)
    return np.concatenate(([np.nan], med))


def line_distance(ball, pitch):
    """Ball distance (m) to the nearest touchline or goal line; negative when outside."""
    length, width = pitch
    x, y = ball[:, 0] * length, ball[:, 1] * width
    return np.minimum.reduce([x, length - x, y, width - y])


def mmss(frame, fps):
    s = frame / float(fps)
    return "{:02d}:{:02d}".format(int(s // 60), int(s % 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", help="only this set-piece Type, e.g. throw-in, corner kick, goals")
    ap.add_argument("--max", type=int, default=None, help="stop after this many figures")
    ap.add_argument("--out", default=os.path.join(ROOT, "inspect"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps, pitch, n = GAMES[GAME]["fps"], GAMES[GAME]["pitch"], GAMES[GAME]["n_video_frames"]
    raw, meta, _ = require(["atd_raw", "atd_meta", "smart_tagging"])

    print("loading tracking (takes a minute)...")
    ball, players, _ = load_atd(raw, meta, n)
    bspeed = ball_speed_ms(ball, pitch, fps)
    pspeed = median_player_speed(players, pitch, fps)
    ldist = line_distance(ball, pitch)

    def low(v):
        return (v or "").strip().lower()

    events = [e for e in load_smart_tagging(fps=fps) if low(e["code"]) in ("set pieces", "goals")]
    if args.type:
        want = low(args.type)
        events = [e for e in events if low(e["tags"].get("Type")) == want or low(e["code"]) == want]
    if args.max:
        events = events[:args.max]
    os.makedirs(args.out, exist_ok=True)

    print("\n{:>3}  {:<22} {:<8} {:>7}  {:>5}  {:>7}".format(
        "#", "type", "team", "video", "dur s", "file"))
    for i, e in enumerate(events):
        kind = low(e["tags"].get("Type") or e["code"])
        a = max(e["start_frame"] - BEFORE_S * fps, 0)
        b = min(e["end_frame"] + AFTER_S * fps, n)
        t = (np.arange(a, b) - e["start_frame"]) / float(fps)  # seconds relative to tag start

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, bspeed[a:b], lw=1)
        missing = np.isnan(ball[a:b, 0])
        axes[0].fill_between(t, 0, 1, where=missing, transform=axes[0].get_xaxis_transform(),
                             color="grey", alpha=0.25, label="ball untracked")
        axes[0].set_ylabel("ball m/s")
        axes[0].set_ylim(0, 30)
        axes[0].legend(loc="upper right")
        axes[1].plot(t, pspeed[a:b], lw=1, color="tab:green")
        axes[1].set_ylabel("median player m/s")
        axes[1].set_ylim(0, 6)
        axes[2].plot(t, ldist[a:b], lw=1, color="tab:red")
        axes[2].axhline(0, color="k", lw=0.5)
        axes[2].set_ylabel("ball to line m")
        axes[2].set_xlabel("seconds from tag start")
        for ax in axes:
            ax.axvspan(0, (e["end_frame"] - e["start_frame"]) / float(fps), color="orange",
                       alpha=0.2)
        fig.suptitle("#{} {} ({}) at video {} -- orange = tag interval".format(
            i, kind, e["tags"].get("Team", "?"), mmss(e["start_frame"], fps)))
        name = "{:03d}_{}.png".format(i, kind.replace(" ", "_").replace("(", "").replace(")", ""))
        fig.savefig(os.path.join(args.out, name), dpi=80)
        plt.close(fig)

        print("{:>3}  {:<22} {:<8} {:>7}  {:>5.1f}  {}".format(
            i, kind[:22], (e["tags"].get("Team") or "?")[:8], mmss(e["start_frame"], fps),
            (e["end_frame"] - e["start_frame"]) / float(fps), name))

    print("\n{} figures in {}".format(len(events), args.out))
    print("For each kind of restart, look for: when the ball stops / leaves the pitch, when players")
    print("slow down, and when the ball is kicked again -- relative to the orange tag interval.")


if __name__ == "__main__":
    main()
