"""Stream 6: broadcast cutaways -- when the TV shows a replay or a close-up, play is usually stopped.

The broadcast recording is not frame-aligned to the tactical one (DATA.md). It does not need to be
for this: both are tracked in pitch coordinates, so the broadcast's ball, shifted by the right
offset, lies on the tactical ball. The offset is found by search, chunk by chunk (30 s), as the
shift that puts the two balls closest (median distance); the live broadcast shares one offset per
half (on BAR-ATH +4015 and +26920 frames, ball distance under 1 m), while replays land elsewhere
or nowhere. No constant here refers to BAR-ATH: the offsets are searched every run.

A tactical frame is a "cutaway" when the broadcast at the matching time is not showing live play:
the camera calibration is invalid (close-up, crowd, graphics), or the broadcast ball disagrees with
the tactical ball for a while (a replay). External evidence (SkillCorner, 20 broadcast-tracked
matches): frames without a camera calibration are dead ball 89-93% of the time, against 37-51% on
average, and cover 67-81% of all dead-ball frames.

    cutaway_mask(tactical_ball, broadcast_ball, broadcast_valid, pitch, fps) -> bool (n_tactical,)
"""

import numpy as np

CHUNK_S = 30
MAX_SHIFT_S = 3600         # search offsets within +-1 hour
MATCH_M = 2.5              # a live chunk: median ball distance under this
REPLAY_M = 6.0             # broadcast ball this far from the tactical ball: not live
REPLAY_SPAN_S = 1.0


def _chunk_offsets(tb, bb, fps):
    """[(chunk_start, offset, median_distance)] by exhaustive search on sampled frames."""
    nt, nb = len(tb), len(bb)
    seen_t = ~np.isnan(tb[:, 0])
    seen_b = ~np.isnan(bb[:, 0])
    chunk = CHUNK_S * fps
    out = []
    lo, hi = -MAX_SHIFT_S * fps, MAX_SHIFT_S * fps
    for c0 in range(0, nb, chunk):
        idx = np.arange(c0, min(nb, c0 + chunk))[::5]
        idx = idx[seen_b[idx]]
        if len(idx) < 20:
            continue
        best = (np.inf, None)
        for step, rng in ((25, (lo, hi)), (1, None)):
            if rng is None:
                if best[1] is None:
                    break
                rng = (best[1] - 25, best[1] + 26)
            for off in range(rng[0], rng[1], step):
                t = idx + off
                ok = (t >= 0) & (t < nt)
                if ok.sum() < 20:
                    continue
                tt = t[ok]
                m = seen_t[tt]
                if m.sum() < 15:
                    continue
                d = np.median(np.hypot(*(bb[idx[ok][m]] - tb[tt[m]]).T))
                if d < best[0]:
                    best = (d, off)
        if best[1] is not None:
            out.append((c0, best[1], best[0]))
    return out


def live_offsets(tb, bb, fps):
    """[(offset, first_b, last_b)]: offsets shared by at least 3 well-matched chunks, each with the
    span of broadcast frames over which it holds (the live broadcast keeps one offset per half)."""
    found = sorted((off, c0) for c0, off, d in _chunk_offsets(tb, bb, fps) if d < MATCH_M)
    groups = []
    for off, c0 in found:
        if not groups or off - groups[-1][-1][0] > fps:
            groups.append([(off, c0)])
        else:
            groups[-1].append((off, c0))
    chunk = CHUNK_S * fps
    return [(int(np.median([o for o, _ in g])), min(c for _, c in g), max(c for _, c in g) + chunk)
            for g in groups if len(g) >= 3]


def cutaway_mask(tactical_ball, broadcast_ball, broadcast_valid, pitch, fps):
    scale = np.asarray(pitch, dtype=float)
    tb = tactical_ball * scale
    bb = broadcast_ball[:len(broadcast_valid)] * scale
    nt, nb = len(tb), len(bb)
    offsets = live_offsets(tb, bb, fps)
    covered = np.zeros(nt, dtype=bool)
    span = max(1, int(REPLAY_SPAN_S * fps))
    for off, first_b, last_b in offsets:
        b = np.arange(nt) - off
        ok = (b >= max(0, first_b - CHUNK_S * fps)) & (b < min(nb, last_b + CHUNK_S * fps))
        t_idx = np.flatnonzero(ok)
        bi = b[ok]
        live = broadcast_valid[bi].astype(bool)
        # replay test: broadcast ball far from the tactical ball, sustained over ~1 s
        d = np.hypot(*(bb[bi] - tb[t_idx]).T)
        far = np.nan_to_num(d, nan=0.0) > REPLAY_M
        far = np.convolve(far.astype(float), np.ones(span) / span, mode="same") > 0.5
        # this offset explains the frame only if the broadcast is live and agrees with it
        covered[t_idx] |= live & ~far
    return ~covered, offsets
