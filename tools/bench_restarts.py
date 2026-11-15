"""Stream 1 (Smart Tagging dead ball + restart team) on EXTERNAL matches, with simulated tags.

    python tools/bench_restarts.py              # needs the Metrica sample data (external_bench.py fetches it)
    python tools/bench_restarts.py --quick      # fewer variants

The Metrica sample games have no Smart Tagging, but their event data says where every restart is
taken. We make tags the way the organisers' starter describes Smart Tagging: a window of ~5 s around
each restart (offsets drawn uniformly, 1.5-3.5 s either side), the restart type from the event,
the team from the event except for throw-ins (Smart Tagging gives none), and 10% of tags dropped
(automatic tagging misses some). Then the full pipeline runs with these tags injected through the
same hooks make_submission.py uses, and is scored with the official scorer.

This measures the mechanism, not the real tagging's quality: run tools/inspect_set_pieces.py on the
real match (locally, figures stay on your machine) to see whether real tags sit around the restart.
Nothing here reads BAR-ATH's ground truth or video.
"""

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
from external_bench import DEFAULT_DIR, build_inputs, fetch, scores  # noqa: E402
from mqh import restarts  # noqa: E402
from mqh.external import metrica  # noqa: E402
from mqh.pipeline import Config, decode, discover_hooks, prepare_arrays  # noqa: E402

KINDS = (("THROW IN", "throw-in"), ("CORNER KICK", "corner kick"), ("GOAL KICK", "goal kick"),
         ("FREE KICK", "free kick"), ("PENALTY", "penalty"))


def _kind(e, first_kick):
    s = e["subtype"].upper()
    if "KICK OFF" in s:
        return "kick off (start)" if e["start"] in first_kick else "kick off (after goal)"
    for key, kind in KINDS:
        if key in s:
            return kind
    return None


def simulated_windows(game, seed, drop=0.10, half_lo=1.5, half_hi=3.5, shift_s=0.0, shape="clip"):
    """[(kind, a, b, team)] tags around every restart of an external game.

    shape "clip" (default): like the real SET PIECES tags, a fixed 35 s window starting 5-10 s
    before the restart (on BAR-ATH 53 of 59 tags are 35.0 s long, and the kick found in the
    tracking sits at 16-29% of the clip, median 20%: a label-free look at the input data). Kick-offs
    stay ~5 s windows. shape "short": the earlier ~5 s window around the restart."""
    rng = np.random.default_rng(1000 + seed)
    first_kick = {first for _, first, _ in metrica.blocks_of(game)}
    out = []
    for e in game["events"]:
        if not metrica._is_restart(e):
            continue
        kind = _kind(e, first_kick)
        u = rng.uniform(half_lo, half_hi, 2)
        dropped = rng.random() < drop
        if kind is None or dropped:
            continue
        r = e["start"] + int(round(shift_s * metrica.FPS))
        team = 0 if kind == "throw-in" else (1 if e["team"] == metrica.HOME else -1)
        if shape == "clip" and not kind.startswith("kick off"):
            a = int(r - rng.uniform(5.0, 10.0) * metrica.FPS)
            out.append((kind, a, a + 35 * metrica.FPS, team))
            continue
        out.append((kind, int(r - u[0] * metrica.FPS), int(r + u[1] * metrica.FPS), team))
    return out


def hooks_for(windows, rcfg, base):
    def dead_hook(n_frames, game, track_dead=None, ball_speed=None):
        stops = restarts.stoppages(windows, n_frames, rcfg, metrica.FPS, track_dead, ball_speed)
        return restarts.dead_from_stoppages(stops, n_frames)

    def team_hook(n_frames, game, track_dead=None, ball_speed=None):
        stops = restarts.stoppages(windows, n_frames, rcfg, metrica.FPS, track_dead, ball_speed)
        return restarts.push_from_stoppages(stops, n_frames, rcfg, metrica.FPS)

    hooks = {k: v for k, v in base.items() if k not in ("restarts", "restart_team")}
    if rcfg is not None:
        hooks["restarts"] = dead_hook
        if rcfg.push > 0:
            hooks["restart_team"] = team_hook
    return hooks


def bench(cases, windows, rcfg, label, base, cfg=Config()):
    rows = []
    for case in cases:
        name, ball, players, team_of_slot, blocks, truths = case
        hooks = hooks_for(windows[name], rcfg, base)
        evidence, dead, blocks, n, touches = prepare_arrays(
            ball, players, team_of_slot, cfg, metrica.PITCH, metrica.FPS, blocks,
            hooks=hooks, game="external")
        labels = decode(evidence, dead, blocks, n, cfg.decoder, touches)
        rows.append({c: scores(t, labels) for c, t in truths.items()})
    out = {}
    for k in ("spell", "macro", "dF1"):
        v = np.array([r["strict"][k] for r in rows])
        out[k] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
    loose = np.mean([r["loose"]["spell"] for r in rows])
    per = " ".join("{:.3f}".format(r["strict"]["spell"]) for r in rows)
    print("{:<44} spellF1 {:.3f} ±{:.3f} (loose {:.3f})  macroF1 {:.3f}  dead F1 {:.3f}   [{}]".format(
        label, *out["spell"], loose, out["macro"][0], out["dF1"][0], per))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shift", type=float, default=0.0,
                        help="move every simulated tag by this many seconds (robustness: real tags "
                             "may not be centred on the restart)")
    args = parser.parse_args()

    root = fetch(args.dir)
    games = [metrica.load_game(n, root) for n in args.games]
    cases = build_inputs(games, args.seeds)
    by_name = {g["name"]: g for g in games}
    windows = {c[0]: simulated_windows(by_name[c[0].rsplit("-s", 1)[0]], int(c[0].rsplit("-s", 1)[1]), shift_s=args.shift)
               for c in cases}
    base = discover_hooks()
    print("stream 1 bench: {} cases, {} simulated tags on average".format(
        len(cases), int(np.mean([len(w) for w in windows.values()]))))

    final = restarts.RestartConfig()                      # what make_submission.py runs
    off = replace(final, push=0.0)
    bench(cases, windows, None, "no tags (external baseline)", base)
    bench(cases, windows, replace(final, detect_kick=False), "v2 (no kick detection), as of 14 Nov", base)
    bench(cases, windows, replace(off, use_tracking=False), "v1: fixed durations, no push", base)
    bench(cases, windows, replace(off, tolerance_s=0.0), "v2: anchored on tracking, clipped to tag", base)
    bench(cases, windows, off, "v2 + tolerance {} s".format(final.tolerance_s), base)
    bench(cases, windows, final, "v2 + tolerance + team push (default)", base)
    if args.quick:
        return
    for t in (0.0, 4.0, 8.0):
        bench(cases, windows, replace(final, tolerance_s=t), "  default, tolerance_s {}".format(t), base)
    for b in (1.0, 5.0):
        bench(cases, windows, replace(final, bridge_s=b), "  default, bridge_s {}".format(b), base)
    for s in (30.0, 90.0):
        bench(cases, windows, replace(final, search_before_s=s), "  default, search_before_s {}".format(s), base)
    for p in (0.25, 1.0):
        bench(cases, windows, replace(final, push=p), "  default, push {}".format(p), base)
    for ps in (1.0, 4.0):
        bench(cases, windows, replace(final, push_s=ps), "  default, push_s {}".format(ps), base)

if __name__ == "__main__":
    main()
