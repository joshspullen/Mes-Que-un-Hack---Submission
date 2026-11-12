# Interfaces between the streams

Every stream returns **one array with one value per video frame** (`n_frames = 199319`, row *i* =
video frame *i*). Pure functions, deterministic, no printing of data, no network. Do not edit
files you do not own; `mqh/pipeline.py` is stream 5's and combines everything.

| stream | file | function | returns |
|---|---|---|---|
| 1 dead ball (Smart Tagging) | `mqh/restarts.py` | `restart_dead_mask(n_frames, game)` | `bool (n_frames,)`, True where the ball is dead |
| 2 dead ball (tracking) | `mqh/dead.py` | `dead_mask(ball, speed, cfg, n_frames, pre_kickoff)` (exists) | `bool (n_frames,)` |
| 3 ball control | `mqh/decode.py` | `team_evidence(ball, players, team_of_slot, pitch, cfg)` (exists) | `float (n_frames,)` in [-1, 1], + home, - away, 0 = no opinion |
| 4 handovers in ball gaps | `mqh/gaps.py` | `gap_evidence(ball, players, team_of_slot, pitch, fps)` | `float (n_frames,)` in [-1, 1], same sign convention; only the frames where the ball is not tracked are used |
| 5 decoder + integration | `mqh/decoder.py`, `mqh/pipeline.py` | `decode_half`, `prepare`, `decode`, `run` | labels `H/A/D/X` |

## How they are combined

```
dead     = dead_mask(...)  OR  restart_dead_mask(...)            # streams 2 and 1
evidence = team_evidence(...)                                    # stream 3
evidence[ball missing] = gap_evidence(...)[ball missing]         # stream 4, clipped to [-1, 1]
labels   = decode_half(evidence, dead, ...) for each half        # stream 5
```

Streams 1 and 4 are **optional plug-ins**: create the file with the function above and the
pipeline uses it automatically (`python make_submission.py` prints `optional streams active: ...`).
If the file does not exist it is skipped. If it exists but returns the wrong shape, type, or
NaN/inf, the run stops with an error naming the function. Return 0 where you have no opinion,
never NaN.

## Conventions

- **Arrays.** `ball` is `(n_frames, 2)` normalised x, y with NaN where the ball is not tracked.
  `players` is `(n_frames, 46, 2)` with NaN for empty slots. `team_of_slot` is `int8 (46,)`,
  0 = home, 1 = away. Multiply by `pitch` (metres) before measuring distances.
- **Frames.** Video frame = ATD `frame_id - 1`. Smart Tagging seconds x 25 = video frame.
- **Constants.** Keep your own constants in your own file (a small dataclass at the top); do not
  add fields to the shared `Config` in `pipeline.py`. Write the reason for every constant in
  `docs/DECISIONS.md`.
- **Rules.** No tuning on this game's ground truth or video. No data in git, in logs you paste
  anywhere, or in any hosted AI tool.

## Testing your stream

- `python tests/test_decoder.py`: unit tests of the decoder, a second.
- `python tests/test_pipeline.py`: the synthetic match end to end, about 40 s. It switches
  auto-discovery off (`hooks={}`) and injects fakes, so your module cannot break it.
- Add your own `tests/test_<stream>.py`. For stream 1, build a tiny made-up Smart Tagging XML in
  the test; for stream 4, use `tests/synth.py` (`turnover_frac=0.4` gives possession changes with
  no dead ball in between, hidden inside ball gaps).
- Real data: run `python make_submission.py --team-name <name>`, then
  `python tools/diagnostics.py submissions/<name>.csv` for per-half sanity numbers.
