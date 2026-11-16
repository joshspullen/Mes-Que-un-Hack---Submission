# Technical document — Mes Que un Hack

*v1, 16 Nov 2026 (engine v4). Rename to `<team_name>.md` (or export to `<team_name>.pdf`) for
submission and fill the three **TODO** fields.*

Team: **TODO team name** · Members: **TODO** · Repository: **TODO public URL**

## 1. Approach

We treat possession as an interval problem, not a per-frame one, because the ranking metric
(`spellF1`) matches whole possessions by temporal overlap. Per frame we estimate two probabilities,
then one decoder turns them into whole possessions and stoppages at once.

- **P(home team has the ball)** (`mqh/team_model.py`). Hand-built signals (nearest player of each
  side, players nearby, ball speed, contested balls; team shape where the ball is not tracked,
  `mqh/gaps.py`; ball touches, `mqh/touches.py`) with their ±1/3/10 s context, fed to a
  gradient-boosted model. Antisymmetric by construction: swapping the teams flips the output
  exactly, so neither side is favoured.
- **P(ball dead)** (`mqh/dead_model.py`). 45 label-free features of the tracking: how much the
  players walk (median speed, share standing still, against the ±5 min median), the shape of both
  teams (spread, centroid, crowding in the box, distance to the centre spot), where the ball was last
  seen (line, corner, out of the pitch), whether it moves, each over several windows before and after
  the frame. Gradient-boosted model.
- **Smart Tagging** (`mqh/restarts.py`, `mqh/v4.py`). For every SET PIECES tag the kick is found in
  the tracking (the ball's first sharp acceleration after being still); the stoppage before it is
  made dead, the first second after it alive, and the restarting team named by the tag gets a push
  on the next two seconds.
- **Joint decoder** (`mqh/decoder3.py`). A three-state Viterbi (home / away / dead) per half, in
  integer arithmetic so the file is identical on every machine. Costs: change of team 10, ball goes
  dead 20, play restarts 20; then dead runs and live spells under 1 s are merged into their
  surroundings. The second-half pre-kick-off gap (frames 110808–113741, FAQ) is forced dead.
  Deciding "dead" and "which team" together lets the team change cheaply across a stoppage, which a
  two-step decoder (dead first, team second) cannot do.

Both models are trained on **external matches only** (section 3) and committed in `models/` with
their hashes; `tools/train_models.py --check` retrains them and confirms the files are
byte-identical.

## 2. Assets used

| asset | used? | how |
|---|---|---|
| Tactical ATD (players + ball) | yes | every feature of both models |
| Smart Tagging | yes (SET PIECES only) | restart kicks and restarting team, as evidence, never as labels |
| Homography | no | the tracking is already in pitch coordinates |
| Broadcast ATD | tried, not used by default | synchronised to the tactical tracking by matching ball paths (`mqh/broadcast.py`), which marks broadcast cutaways (replays, close-ups). On SkillCorner, where cutaways are known, adding them moved dead-ball F1 from 0.916 to 0.919: inside the noise |
| Audio (from the broadcast video) | tried, not used | a whistle detector (`mqh/whistle.py`, 2.8–4.2 kHz tonal energy) finds the kick-offs, but 360 detections on the match and only 14–16% of them near a stoppage boundary: crowd and commentary noise dominate |
| Tactical video | no | — |

## 3. Training data

All learned or tuned parameters come from **30 external matches from three providers**, so no choice
depends on one provider's habits:

| provider | matches | tracking | labels, derived by code |
|---|---|---|---|
| Metrica Sports sample data (github.com/metrica-sports/sample-data, commit `e706dd5`, public with acknowledgement) | 3 | optical, 25 fps | from the event data: dead from a stoppage event to the restart, team of the last controlled action (`mqh/external/metrica.py`) |
| SkillCorner open data (github.com/SkillCorner/opendata, commit `4340d27`, MIT) | 20 A-League 2024/25 | broadcast, 10 Hz → 25 fps | from SkillCorner's phases of play: gap > 2 s = dead (`mqh/external/skillcorner.py`) |
| IDSSE / DFL (Bassek et al. 2025, *Scientific Data* 12, 195, doi:10.1038/s41597-025-04505-y, CC-BY 4.0; data owner DFL) | 7 Bundesliga | TRACAB optical, 25 fps | DFL's own per-frame ball status and team in possession; runs < 1 s merged (`mqh/external/idsse.py`) |

The tracking is degraded to look like ours (`mqh/external/degrade.py`): profile `bar_ath`, set from
label-free aggregates of our tracking (ball missing 41% of frames in long gaps, median 10.7 s;
ball often dropped soon after a stoppage starts), and `match_like`, a label-free calibration
published by another team. The organisers' baseline scores 0.210 on `bar_ath` against 0.2433 on the
real match, so the profile is about as hard as our match. The models are trained on both profiles
(60 cases).

Every choice is scored with the official scorer by **5-fold cross-validation grouped by match**
(`tools/bench_all.py`), and checked provider by provider (`tools/tune_decoder3.py`). No data about
the provided game was produced by hand, and its ground truth and video were never used to train,
tune or select anything.

## 4. Possession rules implemented

- A possession is a maximal run of one team with the ball alive; dead ball is a stoppage, never a
  possession, so "H D H" is two possessions.
- Loose or contested ball: the decoder holds the last team through gaps and weak evidence; a change
  of team costs 10 frames of evidence.
- Dead ball: the dead-ball model, the tagged restarts and the second-half pre-kick-off gap.
- Abstention: no `X` on graded frames (the scorer fills `X` with the previous label for `spellF1`,
  so abstaining gains nothing there and costs `macroF1`).

## 5. Results (external matches, out of fold, `bar_ath` profile, spellF1)

| step | spellF1 |
|---|---|
| organisers' baseline | 0.21 |
| hand-built dead ball + learned team, two-step decoder (v3) | 0.360 |
| + learned dead-ball model | 0.465 |
| + joint H/A/D decoder (v4) | 0.482 |

The decoder gain holds on each provider separately. On the provided match (label-free checks only):
the file is identical on two runs; predicted dead share 47% of graded frames (the models predict
44–46% on IDSSE and SkillCorner, where the truth is 45–47%); 72% of the tagged kicks fall within
2 s of the end of a predicted stoppage. Without the tags only 37% do, which is why the tags stay on
although simulated tags on Metrica did not help (−0.006, inside the noise). The public 5 minutes are
scored once, after the freeze, with `final_check.py`: **TODO** number.

Tried and dropped: self-training on the provided match with confident pseudo-labels (first held-out
check 0.431 → 0.426), broadcast cutaways and whistle (section 2), snapping possession changes onto
touches (inside the noise).

## 6. Limitations

- The ball is untracked on about 41% of graded frames, mostly in long stretches; a handover inside a
  gap is only noticed once the ball returns.
- The degradation is a model of our tracking, not our tracking; the gap between external scores and
  the real match is unknown until the final check.
- External labels follow three conventions, none identical to the organisers'; possession counts on
  the match (194, mean 17 s) depend on that convention.

## 7. Reproduction

```bash
pip install -r requirements.txt            # scikit-learn pinned: the models need the same version
python make_submission.py --team-name <team_name>
python evaluation/validate_submission.py submissions/<team_name>.csv
python -m pytest tests -q
```

Deterministic; the models are committed. To retrain them from scratch (optional, ~10 min, needs the
external data in `data/external/`): `python tools/build_external_cache.py`, then
`python tools/train_models.py --check`. Hash of the submitted file: **TODO** sha256 (printed by
`make_submission.py` and stored in `submissions/<team_name>.manifest.json`).
