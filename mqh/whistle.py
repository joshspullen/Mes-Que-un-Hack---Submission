"""Stream 7: the referee's whistle in the tactical video's audio track.

    whistle_score(wav_path, n_frames, fps) -> float (n_frames,)   one value per video frame
    whistle_events(score, fps)             -> [(start_frame, end_frame)]

The audio is the tactical video's own soundtrack (extracted with ffmpeg, mono 16 kHz), so audio and
video frames line up by construction: the STFT hop is one video frame (640 samples at 16 kHz).

A referee whistle (pea-less, e.g. Fox 40) is a strong, narrow tone around 2.8-4.2 kHz lasting
0.15-1.5 s, much louder in that band than the crowd noise around it. Per frame:
    band ratio  = energy 2.8-4.2 kHz over energy 1.0-2.5 kHz and 5-7 kHz (dB)
    tonality    = peak bin over mean bin inside 2.8-4.2 kHz (dB)
and the score is how far both stand above their own running median (crowd noise changes through
a match). An event is a run of frames above a fixed margin, 0.15-1.5 s long. All constants come
from the physics of a whistle and are fixed a priori; none is fitted on this match (there are no
labels to fit them with). tools/check_whistle.py measures, label-free, how events line up with the
stoppages the tracking finds.
"""

import numpy as np
from scipy.ndimage import median_filter

BAND = (2800.0, 4200.0)
REFS = ((1000.0, 2500.0), (5000.0, 7000.0))
MARGIN_DB = 8.0          # band ratio above its running median
TONAL_DB = 6.0           # peak above the band mean
MIN_S, MAX_S = 0.15, 1.5
BASELINE_S = 30.0


def whistle_score(wav_path, n_frames, fps=25):
    from scipy.io import wavfile

    rate, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32) / 32768.0
    hop = int(round(rate / fps))
    win = 2 * hop
    n = min(n_frames, len(x) // hop)
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    refs = np.zeros_like(band)
    for lo, hi in REFS:
        refs |= (freqs >= lo) & (freqs <= hi)
    window = np.hanning(win).astype(np.float32)
    ratio = np.zeros(n, dtype=np.float32)
    tonal = np.zeros(n, dtype=np.float32)
    step = 4096
    for s in range(0, n, step):
        idx = np.arange(s, min(n, s + step))
        starts = idx * hop - hop // 2
        starts = np.clip(starts, 0, len(x) - win)
        frames = x[starts[:, None] + np.arange(win)[None, :]] * window
        spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
        eb = spec[:, band].mean(axis=1)
        er = spec[:, refs].mean(axis=1)
        ratio[idx] = 10 * np.log10(eb / er)
        tonal[idx] = 10 * np.log10(spec[:, band].max(axis=1) / eb)
    k = int(BASELINE_S * fps) | 1
    base = median_filter(ratio, size=k, mode="nearest")
    out = np.zeros(n_frames, dtype=np.float32)
    out[:n] = np.where(tonal >= TONAL_DB, ratio - base, 0.0)
    return out


def whistle_events(score, fps=25):
    above = score >= MARGIN_DB
    padded = np.concatenate(([0], above.astype(np.int8), [0]))
    d = np.diff(padded)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = [(int(a), int(b)) for a, b in zip(starts, ends)
            if MIN_S * fps <= b - a <= MAX_S * fps]
    return keep
