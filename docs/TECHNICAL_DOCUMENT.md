# Technical document — Mes Que un Hack

*Draft v0, 11 Nov 2026. Sections marked **TODO** need the final method and results. Rename to
`<team_name>.md` (or export to `<team_name>.pdf`) for submission.*

Team: **TODO team name** · Members: **TODO** · Repository: **TODO public URL**

## 1. Approach

We treat possession as an interval problem, not a per-frame one, because the ranking metric
(`spellF1`) matches whole possessions by temporal overlap. A per-frame "nearest player to the ball"
vote flickers and reports many times too many possessions, so the core of the method is a decoder
that needs sustained evidence before it changes team.

- **Evidence.** For every frame with a tracked ball, a signed score: how much closer the nearest
  home player is than the nearest away player (tanh of the distance gap), faded by how far the
  nearest player is from the ball. Frames without a ball score 0, so a tracking gap neither votes nor
  flips the state.
- **Decoder.** A two-state Viterbi (home / away) over the live frames of each half, with a fixed
  switch penalty (v0: 30 frames of consistent evidence). Dead frames are removed first, so
  "H D H" remains two possessions, matching the ground-truth definition.
- **Dead ball.** Rules from the tracking alone (section 4).

*Why this and not something else:* **TODO** — add what the final version does beyond v0 and the
alternatives we tried and dropped.

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

v0 has **no learned components and no training data**; every parameter is a reasoned constant
(`mqh/pipeline.py: Config`). The public 5-minute ground truth was used only to check that frames
and labels line up, never to choose a parameter. **TODO** if later versions learn anything: list
each dataset, its source, who labelled it and how much; none may come from the provided game.

**External data used for one prior (not for fitting).** Metrica Sports sample games 1 and 2
(anonymised event data, https://github.com/metrica-sports/sample-data; its README asks that public
use acknowledges the source). `tools/stoppage_stats.py` measures how often possession changes team
across a restart (throw-in 97%, corner 100%, free kick 27%, all restarts 67%, n = 144). This
supports the sign of one assumption behind `restart_switch_scale`; no constant is fitted to it and
the data is downloaded at run time, not stored in the repository.

## 4. Possession rules implemented

- A possession is a maximal run of one team with the ball alive; dead ball is a stoppage, never a
  possession.
- Loose or contested ball: possession stays with the last team that clearly had it (the decoder
  holds its state through gaps and weak evidence).
- Dead ball (v0, untuned): second-half pre-kickoff gap (frames 110808–113741, per FAQ.md); ball
  tracked outside the pitch (plus a short tail); ball still for 3 s.
- Restarts and set pieces: for every Smart Tagging SET PIECES tag, the tracking dead-ball runs that
  reach the tag (holes under 3 s filled, at most 60 s back) are marked dead up to the restart, the
  restart being where the tracking says play resumed, kept within 2 s of the tag. If the tracking
  sees no stoppage there, a fixed duration per restart type before the tag's middle is used. For 2 s
  after the restart the evidence is pushed (0.5) towards the team named in the tag. Constants
  chosen on external matches with simulated tags (`tools/bench_restarts.py`, `docs/DECISIONS.md`):
  spellF1 0.457 without tags, 0.467 with them, and no loss if the tags are 3 s off.
- Abstention: v0 writes no `X` on graded frames; **TODO** decide whether to abstain inside long ball
  gaps (free on `spellF1`, a miss on `macroF1`).

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
