# Mes Que un Hack — More than a Hack 2027, Phase 1 (ball possession)

Per-frame possession labels (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic Club, ranked on
`spellF1`. Task, data and scoring: the organisers' repository,
https://github.com/metrica-sports/more-than-a-hack-2027 (CHALLENGE.md, DATA.md, EVALUATION.md, RULES.md).

**Deadline: Tuesday 17 November 2026.** Name the files `<team_name>.csv` and `<team_name>.pdf|md`.

## Reproduce the submission

```bash
pip install -r requirements.txt
# unpack the data bundle into ./data (tactical/ and broadcast/), or: export MTAH_DATA_DIR=/path
python make_submission.py --team-name <team_name>
```

This writes `submissions/<team_name>.csv` and runs the official validator on it. The pipeline has no
randomness, so the same data gives the same file. For the final freeze, also run
`pip freeze > requirements-lock.txt` and commit it.

## Check it

```bash
python tests/test_pipeline.py                                  # synthetic match, no data needed
cp submissions/<team_name>.csv evaluation/results/
python evaluation/run_evaluation.py --bootstrap 1000          # public 5 minutes: correctness only
```

The public 5 minutes (17 possessions) are for checking that the pipeline is wired correctly. **Do not
tune on them and do not train on them** (RULES.md).

## How v0 works

1. `mqh/atd.py` — read the tracking onto the video timeline (`frame = frame_id - 1`).
2. `mqh/decode.py` — per frame, a signed score for "who is closer to the ball" (0 when the ball is
   not tracked), then a two-state Viterbi with a switch penalty so a team change needs sustained
   evidence. Gaps in the ball track keep the current possession instead of flickering.
3. `mqh/dead.py` — dead ball from: the 2nd-half pre-kickoff gap (frames 110808–113741), the ball
   outside the pitch, and a ball that stays still for 3 s. **These thresholds are untuned guesses.**
4. `mqh/pipeline.py` — config in one dataclass; graded halves only, `X` elsewhere.

## Rules we must not break

- No hand labels or hand corrections on this game, and no event data or other human-made data about it.
- No data in git (`data/` is ignored; so are caches and frame-level exports). Check `git status` before every push.
- The repo must be public by the deadline and must regenerate the submission file exactly.
- Do not upload the match data or extracts of it to any third-party hosted/cloud LLM (Terms, quoted in FAQ.md).

## Layout

```
make_submission.py   one command: data -> submissions/<team_name>.csv (+ validation)
mqh/                 our code (atd, dead, decode, pipeline)
tests/               synthetic-match test (no real data)
docs/                technical document draft
evaluation/, games.py   organiser files, vendored unmodified (see THIRD_PARTY_NOTICES.md)
```
