# Decision log

Every constant and design choice, who owns it, and why. The match's ground truth and video are
**not** used to choose any of these (RULES.md). "Reasoned" means chosen from the structure of the
task and the metric. Nothing here has been tuned on the real match. This file feeds sections 1, 4
and 6 of the technical document.

## Stream 5: decoder (`mqh/decoder.py`)

| choice | value | why | status |
|---|---|---|---|
| Viterbi, two states (home / away) over live frames | n/a | possessions are intervals; a per-frame vote flickers into many times too many possessions (organisers' baselines) | structural |
| `switch_penalty` | 30 | each frame can swing the score by at most 2, so a run shorter than the penalty can never be supported; 30 frames is about 1.2 s of unanimous evidence, a floor on possession length. Possessions shorter than that rarely match at IoU >= 0.3 anyway | reasoned; lower looked slightly better on clean synthetic data, but real evidence is noisier, so we keep the safer value |
| `restart_switch_scale` | 0.5 | a stoppage ends the previous possession by definition ("H D H" is two possessions), so continuity of the old team is a weaker prior right after one | reasoned; **assumption**: possession often changes team across a stoppage (throw-ins, goal kicks, corners). On synthetic data it helps (+0.05 spellF1) when it always changes, +0.02 at 50%, and is neutral (-0.003) when it never does |
| `restart_min_dead_frames` | 25 (1 s) | shorter dead-mask blips are noise, not a stoppage | reasoned |
| `place_switch_in_gaps` | True | a handover inside a stretch with no informative evidence has no evidence-based position (every position costs the same); we put it at the restart if a stoppage lies in the stretch, otherwise at the midpoint. A uniform prior on the true time makes the midpoint minimise expected boundary error, and boundary error is what temporal IoU punishes | structural; helps in every synthetic world tried (+0.03 to +0.09) because it assumes nothing about who gets the ball |
| `clean_dead_mask`, `min_live_between_dead_frames` | True, 25 (1 s) | "H D H" is two possessions, so one stray dead frame inside play splits a possession and one hole inside a stoppage invents one. We drop dead runs shorter than `restart_min_dead_frames` and fill live bursts under 1 s between two stoppages | reasoned; protects the decoder from noisy masks of streams 1 and 2 |
| integer arithmetic (`SCALE` = 10,000) | n/a | the repo must regenerate the exact submitted file; evidence is rounded to 1/10,000 and the Viterbi runs on integers, so BLAS or library differences cannot flip a label. Ties prefer staying | structural; checked by `test_integer_decoding_ignores_float_noise` and the golden hash |
| `info_min` | 0.05 | below this the vote is nearly a tie and carries no information | reasoned |
| no minimum-possession merge | n/a | the penalty already sets a floor on run length (about 30 frames), so a separate merge step adds moving parts for almost no effect | decided |
| abstention (`X`) on graded frames | never | the scorer fills an `X` with the previous label before matching possessions, so inside a gap `X` and "hold the state" score the same on `spellF1`, while `X` costs on `macroF1` (the tie-break). At a handover `X` is worse, because the old label carries over | decided |
| `X` outside the two graded halves | always | never scored | given |

### Ideas taken from the team plan (competitors' write-ups), stream 5 only

Source: `Plan de projet — Hackathon FCB possession (5 rôles)`, which summarises ZesterJ, Letinsh,
TrymBardsen and Barca-Hackathon / ZainnB. Their scores are reported second-hand and were not
reproduced here.

| idea | decision | why |
|---|---|---|
| snap each H<->A change onto the new team's first touch within +-1 s (R5 item 3, "8 frames late costs 11 points") | **built, off by default** (`mqh/refine.py`, hook `mqh/touches.py`, `snap_window_frames` = 0) | needs a touch detector from stream 3 that does not exist yet. On synthetic matches it is neutral (spellF1 0.842 -> 0.842 and 0.826 -> 0.826 on two seeds; with deliberately noisy touches boundary error got slightly worse), so it only goes on if the external bench shows a gain beyond the noise |
| tune constants by directly maximising spellF1 on external matches, leave-one-match-out, with the official scorer; keep the simplest setting within the noise band | **built** (`tools/tune_decoder.py`) | needs external cases (stream 2 of the plan). Works now on synthetic cases to show the procedure; the real numbers must come from external matches degraded like our tracking |
| ablation table, baseline to full | **built into the same tool** | the tool prints plain Viterbi, + mask cleaning, + place in gaps, + cheaper restart per case |
| X only outside the graded halves | already ours | the scorer fills X with the previous label, so it only costs macroF1 |
| minimum durations on dead and live phases (state expansion) | already ours, as `clean_dead_mask` (1 s dead, 1 s live burst) and the switch penalty | a hard minimum possession `m` is covered by the penalty; adding it would double the knobs |
| two-stage decoding with a soft `p_dead` (Viterbi on log p_dead first) | not built | our dead streams give hard masks. `viterbi_two_state` already accepts a per-step penalty, so this is a ten-line change the day a calibrated `p_dead` exists |
| calibrated `p_home` from a learned control model | not built | no learned model exists. The decoder takes a signed score, `p_home` would enter as `logit(p)/2` clipped to [-1, 1], and the penalty would need re-deriving |
| HSMM with typical possession and stoppage durations | not built | durations we could estimate externally would mostly restate the penalty; unverifiable until the external bench exists |
| self-training on BAR-ATH pseudo-labels, 5-seed ensembles, domain weighting | not stream 5 | they act on learned models (streams 3 and 4 of the plan); v0 has none. The plan itself says to validate self-training on an external match first |
| a script that alone reads the public ground truth, once, after the freeze | adopted as a rule | see `docs/SUBMISSION_CHECKLIST.md`, freeze protocol |

### External evidence on stoppages (`tools/stoppage_stats.py`)

The restart discount rests on the assumption that possession often changes team across a stoppage.
We measured it on **external** open data: Metrica Sports sample games 1 and 2 (public repository
`metrica-sports/sample-data`; its README asks users to acknowledge the source, which we do here and
in the technical document). Not the hackathon match. For each restart we compared the restarting
team with the team that last had the ball in open play (last pass, shot or recovery).

| restart | n | team changed | median gap to restart |
|---|---|---|---|
| throw-in | 60 | 97% | 10 s |
| corner | 18 | 100% | 30 s |
| free kick | 62 | 27% | 22 s |
| all restarts | 144 | 67% | 18 s |

Reading it: the real world sits between the "50%" and "100%" rows of the synthetic table below,
where the default (restart discount on) is the best or tied variant, so the constant is supported.
Caveats: two games, one competition, crude "last holder" definition; the gap includes the ball
travelling out and silent events, so it overstates the dead time. We use this to confirm the sign
of an assumption, not to fit a number. `restart_switch_scale` stays at the reasoned 0.5.

### Analysed and not implemented

| idea | why not |
|---|---|
| posterior boundaries (forward-backward instead of Viterbi) | with a two-state chain and a constant switch cost it reduces to the same boundaries as `place_switches`; extra code, no new information |
| a joint "dead" state inside the HSMM | the dead mask comes from streams 1 and 2 with their own evidence; fusing it inside the decoder couples the streams and hides which one is wrong |
| explicit-duration HSMM | the penalty already floors run length, and we have no external length model worth the moving parts |
| ensemble of several penalties | the score is flat for 10 to 45 (table below), so there is nothing to average |
| shifting labels by a time offset to align with the reference | this would be calibrating on the ground truth or video, which the rules forbid |

### Synthetic robustness check (`tools/sensitivity_synth.py`)

Mean `spellF1` over three made-up matches with known possessions, 40% of possession changes with no
dead ball, ball hidden 40% of the time. Mechanism check only; the real match is noisier and the
numbers must not be read as expected scores.

| team changes across a stoppage | variant | pen 10 | 20 | 30 | 45 | 60 | 90 |
|---|---|---|---|---|---|---|---|
| 100% | plain Viterbi | .711 | .711 | .708 | .709 | .704 | .689 |
| | + place in gaps | .803 | .792 | .788 | .775 | .759 | .733 |
| | + cheaper restart (default) | .853 | .844 | .839 | .829 | .810 | .783 |
| 50% | plain Viterbi | .731 | .729 | .731 | .726 | .721 | .707 |
| | + place in gaps | .804 | .794 | .790 | .781 | .772 | .751 |
| | + cheaper restart (default) | .828 | .820 | .814 | .803 | .789 | .777 |
| 0% | plain Viterbi | .752 | .749 | .751 | .749 | .746 | .739 |
| | + place in gaps | .786 | .782 | .778 | .775 | .770 | .763 |
| | + cheaper restart (default) | .785 | .780 | .776 | .771 | .764 | .752 |

Reading it: the score is flat for penalties 10 to 45 and falls slowly beyond; placing handovers in
gaps always helps; the restart discount helps or is neutral across the whole range of assumptions.

## Streams 2 and 3: settings currently in `Config` (`mqh/pipeline.py`)

| constant | value | why | status |
|---|---|---|---|
| `margin_scale_m` | 2.0 | nearest-player distance gap at which the vote becomes decisive (tanh(1) = 0.76) | reasoned, owner may change |
| `proximity_scale_m` | 6.0 | how fast the vote fades when nobody is near the ball | reasoned |
| dead ball: out of bounds / still ball / 2nd-half pre-kickoff | see `Config` | pre-kickoff gap (frames 110808-113741) is published in FAQ.md; the others are hypotheses and **almost never fire on the real tracking** | **untuned, weak** |

Add a row here for every constant you introduce, with the reason. "It looked better on the public
5 minutes" is not a reason and is against the rules.

## Stream 3: `EvidenceConfig` (`mqh/decode.py`)

`team_evidence` = vote x w_near x w_speed x w_contest (formula in the docstring). Its constants live
in `EvidenceConfig`. `margin_scale_m` and `proximity_scale_m` are also still in `Config` above;
when `team_evidence` is given `Config`, those two values are taken from it and the rest from
`EvidenceConfig` defaults, so the two places cannot disagree silently. None of these values were
tuned on this match.

| constant | value | why | status |
|---|---|---|---|
| `margin_share` | 0.6 | mix of the nearest-player margin vote and the nearby-player count vote; the margin stays the main signal, the count breaks near-ties when several players are around the ball | reasoned |
| `support_radius_m` | 4.0 | Gaussian radius for the soft count of each team's players around the ball: about two strides, the distance from which a player can reach the ball within a second | reasoned |
| `support_eps` | 0.5 | pseudo-count in the count vote, so a single distant player does not give a full +-1 vote | reasoned |
| `speed_mid_ms` | 10.0 | ball speed at which the vote is halved: dribbles stay well below, passes and shots are well above, so a flying ball mostly does not vote for whoever it passes near | reasoned |
| `speed_width_ms` | 2.0 | softness of that cut, so tracking noise around 10 m/s does not flip the weight | reasoned |
| `speed_floor` | 0.2 | a flying ball keeps a fifth of its vote; exact zeros right before a stoppage let the decoder park a switch there and create 3-frame fake spells | reasoned; to recheck with stream 5's gap placement, which puts handovers inside stretches without informative evidence |
| `speed_half_window` | 3 | ball speed from a centred difference over +-3 frames (0.12 s each side at 25 fps); a one-frame difference turns ~0.3 m of jitter into ~10 m/s and every dribble would look like a pass (`test_jitter_is_not_a_pass`) | reasoned |
| `contest_radius_m` | 1.5 | both teams this close to the ball counts as a duel | reasoned |
| `contest_floor` | 0.35 | vote multiplier for a fully contested ball: weaker, not zero, so a duel neither drives nor blocks a switch on its own | reasoned |

## External bench, dead ball v2, ball-gap evidence, touches (13 Nov)

Everything below was chosen on **external** matches only: Metrica Sports' three public sample games
(https://github.com/metrica-sports/sample-data, acknowledged here and in the technical document),
labels derived from their event data by code (`mqh/external/metrica.py`, two readings of "control":
`strict` and `loose`), tracking degraded to look like ours (`mqh/external/degrade.py`). BAR-ATH's
ground truth and video were not read. Commands: `python tools/external_bench.py`,
`python tools/fit_dead.py`, `python tools/fit_gaps.py`, `python tools/check_touches.py`,
`python tools/tune_decoder.py --cases .cache/external_cases`.

### Ablation on the external bench (6 cases = 3 games x 2 degradation seeds, strict labels)

| pipeline | spellF1 | macroF1 | dead-ball F1 |
|---|---|---|---|
| organisers' baseline (nearest player, +-50 vote) | 0.329 | 0.740 | 0.747* |
| team branch before this work (v0 dead rules, penalty 30) | 0.308 ±0.011 | 0.465 | 0.056 |
| + dead ball v2 (stream 2) | 0.387 ±0.008 | 0.759 | 0.766 |
| + switch penalty 10 (stream 5, tuned leave-one-match-out) | 0.436 ±0.004 | 0.757 | 0.766 |
| + ball-gap evidence (stream 4) and touches (stream 3b) | **0.457 ±0.009** | **0.769** | 0.766 |

\* The baseline writes `D` whenever the ball is unseen, and Metrica's sample tracking hides the ball
on 90-99% of dead frames, so its dead-ball F1 here is flattering. On BAR-ATH it scores 0.2433.
Absolute numbers are against event-derived labels, not the organisers' convention: use them to
compare variants, not to predict the final score.

### Stream 2 v2 (`mqh/dead.py`: `DeadConfig`, `tracking_dead_mask`)

| choice | value | why | status |
|---|---|---|---|
| model | logistic on 4 features | players walk at a stoppage (`walk_abs`, `walk_long`), the ball was last seen near a line (`near_line`), a moving ball is live (`ball_moving`) | fitted on external games, leave-one-game-out AUC 0.92 / 0.92 / 0.92 |
| `no_ball`, `gap_log` | weight 0 | Metrica's sample data hides the ball at stoppages; these features learn that provider habit. Without them AUC barely moves. `tools/sim_check.py` measures whether BAR-ATH hides it too (inside vs outside set-piece tags); if so, refit with them | decided |
| `walk_ratio` | weight 0 | collinear with `walk_long` (weights +8.4 / -7.4 together) | decided |
| speeds over ~1 s (players) and +-0.2 s (ball) | `speed_span_s` 1.0 | frame-to-frame speeds turn tracking jitter into running: under heavy simulated jitter dead-ball F1 fell from 0.77 to 0.00; with these spans it stays 0.76 | measured |
| `threshold` | 0.5 | 0.4 / 0.5 / 0.6 give 0.411 / 0.403 / 0.375 spellF1 (leave-one-game-out); 0.4 vs 0.5 is inside the noise, so the neutral value stays | measured |
| `Config.dead` | `DeadConfig()` | `None` switches back to the v0 rules (used by `tests/test_pipeline.py`, whose synthetic players do not slow down at stoppages) | default |

### Stream 5 (`mqh/decoder.py`)

| choice | value | why | status |
|---|---|---|---|
| `switch_penalty` | 30 -> **10** | `tune_decoder.py` on 12 external cases, whole matches held out: chosen-on-the-others 0.457 vs default 0.419, gain above the noise (0.006). Checked under lighter and heavier degradation: 10 and 20 are within noise of each other, 30 is worse in both | tuned on external data |
| `restart_min_dead_frames`, `restart_switch_scale` | unchanged | the tuning keeps them | confirmed |
| `snap_window_frames` | 0 (unchanged) | with `mqh/touches.py`, windows 6 / 12 / 25 frames gain +0.001 to +0.002 spellF1, inside the noise | measured, kept off |
| `tune_decoder.py` | holds out every case of a match together; reads per-case halves | several seeds and label readings per match would otherwise leak into the choice | fixed |

### Stream 4 (`mqh/gaps.py`) and stream 3b (`mqh/touches.py`)

| choice | value | why | status |
|---|---|---|---|
| gap evidence | width, depth and push differences, logistic without intercept | the team with the ball spreads out and moves forward; antisymmetric so no side is favoured. AUC "home has the ball" 0.83-0.87 on live frames, 0.61-0.67 inside ball gaps | fitted on external games (`tools/fit_gaps.py`) |
| `GapConfig.weight` | 0.5 | weights 0.25 / 0.5 / 1.0 give 0.475 / 0.476 / 0.475 spellF1 vs 0.459 without; the middle value | measured |
| touches | velocity change >= 6 m/s over +-0.12 s with a player within 2 m | precision 0.39-0.50 / recall 0.81-0.83 against Metrica pass starts and receptions (event lists miss many touches, so precision is a lower bound) | checked on external events |

## Stream 1: Smart Tagging dead ball v2 and restart team (15 Nov)

Chosen on **external** matches only, with **simulated** tags: Metrica's sample games have no Smart
Tagging, so `tools/bench_restarts.py` builds tags from their event data the way the organisers'
starter describes the real ones (a ~5 s window around each restart, team given except for
throw-ins, 10% of tags dropped). `--shift S` moves every tag by S seconds, to see what happens if
the real tags are not centred on the restart. Reproduce: `python tools/bench_restarts.py`.

| pipeline (6 cases, strict labels, spellF1) | tags centred | tags 3 s early | tags 3 s late |
|---|---|---|---|
| no tags | 0.457 | 0.457 | 0.457 |
| v1: fixed seconds before the tag's middle (what the branch had) | 0.456 | 0.437 | 0.412 |
| v2: stoppage taken from the tracking dead runs near the tag, restart clipped into the tag | 0.462 | 0.449 | 0.445 |
| v2 + `tolerance_s` 2 | 0.460 | 0.449 | 0.454 |
| v2 + tolerance + restart-team push (**default**) | **0.467** | 0.455 | 0.460 |

Standard error between cases is about ±0.009, so read the differences per case: the default beats
"no tags" in all 6 cases when the tags are centred (+0.004 to +0.013), and is level with it
(-0.002 / +0.003 on average) when the tags are 3 s off. v1 is never better than no tags and loses
0.02 to 0.045 when the tags are off, which is why it is no longer the default.

| constant (`RestartConfig`) | value | why | status |
|---|---|---|---|
| `use_tracking` | True | the tracking already finds most stoppages (dead-ball F1 barely moves); what the tag adds is where the stoppage ends and which pieces belong together | measured |
| `tolerance_s` | 2.0 | 0 is best with centred tags (0.468) but costs 0.010 when tags are 3 s late; 2 costs 0.001 centred and is the best under a shift; 4 and 8 are slightly worse everywhere | measured, chosen for robustness |
| `bridge_s` | 3.0 | 1 / 3 / 5 give 0.468 / 0.467 / 0.464: flat | measured, kept |
| `search_before_s` | 60 | 30 / 60 / 90 give 0.466 / 0.467 / 0.467: flat | measured, kept |
| `push`, `push_s` | 0.5, 2.0 s | strength 0.25 / 0.5 / 1.0 is flat (0.466 / 0.467 / 0.467); duration 1 / 2 / 4 s gives 0.463 / 0.467 / 0.463 | measured |
| `pre_dead_s` | per type | v1 guesses, now only the fallback when the tracking sees no stoppage near a tag | reasoned, unchanged |
| `HOME_NAMES`, `AWAY_NAMES` | Barcelona / Athletic | Team tag text to side, by substring; an unmatched name gives no push | **check** the Team values printed by `python starter/load_smart_tagging.py` |

Limits: the tags are simulated, so this validates the mechanism, not the real tagging. Whether the
real tags sit around the restart is what `tools/inspect_set_pieces.py` shows (tracking only, no
ground truth). The gain is small (+0.010 at best) and its worst measured case is neutral.

## Learned team evidence (stream 3c) and the match-like profile (15 Nov)

External only, as above. Reproduce: `python tools/external_bench.py --profile match_like --learned`
(and without `--profile` for the default one).

**Match-like degradation profile** (`mqh/external/degrade.py: MATCH_LIKE`). Parameters taken from a
label-free calibration published by another team (MikolajEkiert/more-than-a-hack-2027,
docs/DECISIONS.md, 14 Nov): 25% of live frames lose the ball, ball offset ~2 m, the ball stays
visible after 90% of stoppages for 1-6 s. Check: the organisers' baseline (smooth50) scores 0.216
on it, against 0.2433 published for the real match (0.329 on the default profile, too easy).

**Where the points are** (oracle test, match-like, 6 cases): our pipeline 0.360; with perfect dead
ball 0.433 (+0.07); with team evidence 60% perfect 0.799 (+0.44). Team evidence is the bottleneck.

| pipeline (spellF1, strict labels, mean of 6 cases) | default profile | match-like profile |
|---|---|---|
| organisers' baseline smooth50 | 0.329 | 0.216 |
| hand-built evidence (13 Nov state + stream 1 v2) | 0.457 ±0.009 | 0.360 ±0.010 |
| + learned team evidence, leave-one-game-out | **0.545 ±0.012** | **0.436 ±0.008** |

All 6 cases improve under both profiles. Team accuracy on live frames: ~0.67 -> ~0.86.

| choice | value | why | status |
|---|---|---|---|
| model | HistGradientBoosting, 200 trees, lr 0.08, 31 leaves, l2 1.0, no early stopping | logistic regression on the same features gives 0.389 (match-like), boosted trees 0.426; settings are sklearn-like defaults, not tuned | measured |
| features | 15 antisymmetric (distance / count differences, stream-3 score, team shape, last / next touch team, ±1/3/10 s context) + 5 symmetric (ball speed, ball seen, nearest distance, time since / until a touch) | what stream 3, 3b and 4 already compute, in context | reasoned |
| team swap | every training frame also shown swapped; prediction = mean of the frame and its swap | evidence(swap) = -evidence exactly (tested), so neither side is favoured | by construction |
| training data | 3 Metrica games x 2 profiles x 2 seeds, `strict` labels, every 5th live frame | both profiles so the model does not depend on one guess of our tracking | decided |
| run time | trained at run time from the pinned sample-data commit, cached in `.cache/` keyed by the code and sklearn version | deterministic: single thread, seed 0; two fits give the same evidence hash | checked |
| `switch_penalty` with it | stays 10 | `tune_decoder.py` on the learned cases: default inside the noise under both profiles | confirmed |
| `make_submission.py` | learned team on by default, `--no-learned-team` to switch off | `Config.learned_team` stays False so tests and fits need no download | decided |

## First run on the real tracking, stream 1 v3, BAR-ATH profile (15 Nov, evening)

Label-free statistics of BAR-ATH's tracking (`tools/sim_check.py`; no ground truth, no video):
ball missing 40.7% of graded frames, in few long gaps (1.4 per minute, median 10.7 s, p90 34 s);
missing 32% inside set-piece tags vs 44% elsewhere (so "no ball" is NOT a stoppage signal here:
stream 2 was right to leave `no_ball` / `gap_log` out); player positions very smooth (second
difference 0.001 m); stream 2 finds 25.5% dead ball.

**Smart Tagging shape** (`tools/inspect_set_pieces.py` + an automatic kick search in the tracking):
53 of the 59 SET PIECES tags are fixed 35.0 s clips (kick-offs 5 s); the kick sits at 16-29% of
the clip (median 20%, about 7 s in), not in the middle. Stream 1 v2 anchored on the middle.

| choice | value | why | status |
|---|---|---|---|
| stream 1 v3: `RestartConfig.detect_kick` | True | the restart is found inside each tag by an algorithm (first kick > 6 m/s after >= 1 s of still or unseen ball), the stoppage runs back from there along the tracking's dead runs; no kick found -> v2 rule. External bench with tags shaped like the real ones (35 s, starting 5-10 s before the restart): no tags 0.457, v2 0.425 (tags hurt), **v3 0.476** | measured (`tools/bench_restarts.py`, tags now `shape="clip"`) |
| degradation profile `bar_ath` | ball missing 30% of live frames in long gaps (median 10 s), short source gaps interpolated, smooth players, ball offset 2 m | reproduces the statistics above on the external games (41-46% missing, 1.6 gaps/min, median 9-13 s); organisers' baseline scores 0.210 on it vs 0.2433 published | calibrated label-free |
| learned team model profiles | default + match_like + bar_ath | on the bar_ath profile, leave-one-game-out: no model 0.362, model trained on two profiles 0.408, on all three **0.423**; match-like 0.436 -> 0.431 (noise) | measured |
| dead-ball weights refitted on the three profiles | not adopted | dead F1 up (0.705 -> 0.727 on bar_ath) but spellF1 down on every profile (-0.005 to -0.012) | measured, kept the 13 Nov weights |
| decoder on bar_ath cases | defaults kept | `tune_decoder.py`: chosen-on-the-others = default (0.411) | confirmed |

**Real run** (`make_submission.py`, twice): identical SHA-256; 335 possessions over the two halves
(mean 13 s, median 7-8 s), 30% dead ball. Format valid. The public 5 minutes were not scored.
