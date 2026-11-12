"""Stream 5: snap each team change onto the new team's first touch.

A possession starts when the new team touches the ball. The Viterbi needs sustained evidence before
it switches, so its boundary tends to lag the real handover by a fraction of a second, and temporal
IoU punishes boundary error. If a touch detector (stream 3, hook `mqh/touches.py`) says the new team
touched the ball close to a decoded H<->A change, the change is moved onto that touch.

Only direct H<->A changes move. Dead-ball edges belong to the dead-ball streams, a change never
crosses another change or a dead frame, and with no touch inside the window nothing moves.
"""

import numpy as np

DEFAULT_WINDOW = 25  # frames (1 s): about the lag the switch penalty adds, see docs/DECISIONS.md


def snap_switches(labels, touch_frames, touch_teams, window=DEFAULT_WINDOW):
    """Return a copy of `labels` (one half, 'H'/'A'/'D') with H<->A changes moved onto touches.

    touch_frames: sorted-or-not int frame indices *relative to labels* (0 = first label)
    touch_teams:  0 = home, 1 = away, same length
    """
    out = np.array(labels, copy=True)
    n = len(out)
    if window <= 0 or n < 2 or len(touch_frames) == 0:
        return out
    frames = np.asarray(touch_frames, dtype=np.int64)
    teams = np.asarray(touch_teams, dtype=np.int64)
    order = np.argsort(frames, kind="stable")
    frames, teams = frames[order], teams[order]

    change = np.flatnonzero((out[1:] != out[:-1]) & np.isin(out[1:], ["H", "A"])
                            & np.isin(out[:-1], ["H", "A"])) + 1
    for s in change:
        new, old = out[s], out[s - 1]
        want = 0 if new == "H" else 1
        # the run before s starts after the previous change or dead frame; the run after s ends
        # before the next change or dead frame
        lo = s
        while lo > 0 and out[lo - 1] == old:
            lo -= 1
        hi = s
        while hi < n - 1 and out[hi + 1] == new:
            hi += 1
        # keep at least one frame on each side
        a, b = lo + 1, hi
        sel = (teams == want) & (frames >= max(a, s - window)) & (frames <= min(b, s + window))
        if not sel.any():
            continue
        cands = frames[sel]
        t = int(cands[np.argmin(np.abs(cands - s))])    # ties: the earlier touch
        if t < s:
            out[t:s] = new
        elif t > s:
            out[s:t] = old
    return out
