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
| `info_min` | 0.05 | below this the vote is nearly a tie and carries no information | reasoned |
| no minimum-possession merge | n/a | the penalty already sets a floor on run length (about 30 frames), so a separate merge step adds moving parts for almost no effect | decided |
| abstention (`X`) on graded frames | never | the scorer fills an `X` with the previous label before matching possessions, so inside a gap `X` and "hold the state" score the same on `spellF1`, while `X` costs on `macroF1` (the tie-break). At a handover `X` is worse, because the old label carries over | decided |
| `X` outside the two graded halves | always | never scored | given |

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
