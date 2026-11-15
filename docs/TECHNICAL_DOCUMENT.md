# Technical document — Mes Que un Hack

*Draft v0, 11 Nov 2026. Sections marked **TODO** need the final method and results. Rename to
`<team_name>.md` (or export to `<team_name>.pdf`) for submission.*

Team: **TODO team name** · Members: **TODO** · Repository: **TODO public URL**

## 1. Approach

We treat possession as an interval problem, not a per-frame one, because the ranking metric
(`spellF1`) matches whole possessions by temporal overlap. Per frame we estimate two things, then a
decoder turns them into whole possessions:

- **Is the ball dead?** (stream 2, `mqh/dead.py`) A logistic model on four label-free signals of the
  tracking: players walk at a stoppage (median speed over ~1 s, absolute and against its own
  ±5 min median), the ball was last seen within 2 m of a line, a moving ball is live. Weights fitted
  on external matches only. Smart Tagging set pieces then fix where each stoppage ends and which
  team restarts (stream 1, `mqh/restarts.py`).
- **Which team has it?** (stream 3) A hand-built score (nearest player of each side, players nearby,
  ball speed, contested balls), a team-shape vote where the ball is not tracked (stream 4,
  `mqh/gaps.py`: the team in possession spreads out and pushes forward), ball touches
  (`mqh/touches.py`), all fed with their context (±1/3/10 s) into a gradient-boosted model trained
  on external matches (stream 3c, `mqh/team_model.py`). It is antisymmetric by construction:
  swapping the teams flips the evidence exactly, so neither side is favoured.
- **Decoder** (stream 5, `mqh/decoder.py`): a two-state Viterbi (home / away) over the live frames
  of each half, integer arithmetic so the file is identical on every machine, switch penalty 10
  frames, an easier change of team across a restart. Dead frames are removed first, so "H D H" is
  two possessions, as the ground truth defines them.

*Why this:* on degraded external matches (section 3), the dead-ball model took dead-ball F1 from
0.06 to 0.77 and the learned team evidence took team accuracy on live frames from ~67% to ~86%;
an oracle test showed team evidence was the larger bottleneck (perfect team evidence +0.44 spellF1,
perfect dead ball +0.07). Tried and dropped: snapping possession changes onto touches (gain inside
the noise), ball-missing features for dead ball (they learn a habit of one provider's tracking),
heavier sequence networks (no GPU, little gain reported by other teams over boosted trees).

## 2. Assets used

| asset | used? | how |
|---|---|---|
| Tactical ATD (tracking, players + ball) | yes | evidence, dead-ball rules |
| Smart Tagging | yes (SET PIECES tags only) | each tagged restart (throw-in, goal kick, corner, free kick, penalty, kick-off after a goal) marks where a stoppage ends and, except for throw-ins, which team restarts. Used as automatic evidence, never as a label |
| Homography | no | the tracking is already in pitch coordinates, so no pixel conversion is needed |
| Tactical / broadcast video | no | not processed by any part of the pipeline |
| Broadcast ATD and homography | no | not frame-aligned to the tactical video; mapping back was out of scope |

Say explicitly which of these we tried and dropped, and why.

## 3. Training data

All learned or tuned parameters come from **external matches only**: Metrica Sports' three public
sample games (https://github.com/metrica-sports/sample-data, pinned commit `e706dd5`; anonymised
tracking and event data, released by Metrica for public use with acknowledgement). Labels are
derived from their event data **by code** (`mqh/external/metrica.py`): dead from a stoppage event
(ball out, foul, goal, offside) to the next restart, possession to the team of the last controlled
action (two readings: `strict`, used for training, and `loose`, used as a check). The tracking is
degraded to look like ours (`mqh/external/degrade.py`, two profiles: `default`, set from label-free
aggregates of our tracking, and `match_like`, a label-free calibration published by another team:
ball lost on 25% of live frames, ball off by ~2 m, ball still visible after 90% of stoppages). The
organisers' baseline scores 0.216 under `match_like` against 0.2433 published for the real match,
so that profile is about as hard as our match.

Every choice is scored leave-one-game-out (a model or setting chosen on two games, scored on the
third) with the official scorer: `tools/external_bench.py`, `tools/fit_dead.py`,
`tools/fit_gaps.py`, `tools/tune_decoder.py`. The learned team model is retrained at run time from
the pinned data (deterministic: single thread, fixed seed; same file twice). No data about the
provided game was produced by hand, and its ground truth (public 5 minutes) and video were never
used to train, tune or select anything.

## 4. Possession rules implemented

- A possession is a maximal run of one team with the ball alive; dead ball is a stoppage, never a
  possession.
- Loose or contested ball: possession stays with the last team that clearly had it (the decoder
  holds its state through gaps and weak evidence).
- Dead ball: the second-half pre-kickoff gap (frames 110808–113741, per FAQ.md), plus the
  stream-2 model (players walking, ball last seen near a line, ball not moving) thresholded at 0.5.
- Restarts and set pieces: for every Smart Tagging SET PIECES tag, the tracking dead-ball runs that
  reach the tag (holes under 3 s filled, at most 60 s back) are marked dead up to the restart, the
  restart being where the tracking says play resumed, kept within 2 s of the tag. If the tracking
  sees no stoppage there, a fixed duration per restart type before the tag's middle is used. For 2 s
  after the restart the evidence is pushed (0.5) towards the team named in the tag. Constants
  chosen on external matches with simulated tags (`tools/bench_restarts.py`, `docs/DECISIONS.md`):
  spellF1 0.457 without tags, 0.467 with them, and no loss if the tags are 3 s off.
- Abstention: no `X` on graded frames. The scorer fills `X` with the previous label for `spellF1`,
  so abstaining gains nothing there and costs `macroF1`.

## 5. Results

**TODO** final numbers on the public 5 minutes (correctness check only, with bootstrap intervals),
possession counts and durations against the baselines, and a failure-case figure. A synthetic-match
test (`tests/test_pipeline.py`) verifies the plumbing: there the decoder reports about the true
number of possessions (305 vs 284) while the same evidence without the switch penalty reports 848.

## 6. Limitations

- The ball is untracked on about 41% of graded frames (DATA.md), mostly in long stretches; a
  handover inside a gap is only noticed once the ball returns.
- Dead-ball rules are heuristics and have not been calibrated.
- Nearest-player is a weak proxy for control (passes, deflections, aerial balls).
- **TODO** what we would fix with more time.

## 7. Reproduction

```bash
pip install -r requirements.txt
python make_submission.py --team-name <team_name>
python evaluation/validate_submission.py submissions/<team_name>.csv
```

Deterministic; no randomness. Check with `python tools/repro_check.py` (runs the pipeline twice and
compares SHA-256 hashes). Hash of the submitted file: **TODO** sha256. Verified on a fresh clone:
**TODO** date and who.

Environment: pinned in `requirements-lock.txt` (**TODO** generate with `pip freeze` at the freeze).
