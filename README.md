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
`pip freeze > requirements-lock.txt` and commit it. It also writes `submissions/<team_name>.manifest.json`
(sha256 of the CSV, config, git commit, versions); `python tools/repro_check.py` regenerates the file and
compares hashes.

## Check it

```bash
python tests/test_decoder.py                                   # decoder unit tests, ~1 s
python tests/test_pipeline.py                                  # synthetic match, no data needed, ~40 s
python tests/test_golden.py                                    # fixed made-up match must hash to tests/golden.json
python tools/diagnostics.py submissions/<team_name>.csv        # per-half sanity numbers
python tools/stoppage_stats.py                                 # external-data stoppage rates (downloads Metrica samples)
cp submissions/<team_name>.csv evaluation/results/
python evaluation/run_evaluation.py --bootstrap 1000          # public 5 minutes: correctness only
```

The public 5 minutes (17 possessions) are for checking that the pipeline is wired correctly. **Do not
tune on them and do not train on them** (RULES.md).

## How v0 works

1. `mqh/atd.py` — read the tracking onto the video timeline (`frame = frame_id - 1`).
2. `mqh/decode.py` — per frame, a signed score for "who is closer to the ball" (0 when the ball is
   not tracked, so a tracking gap casts no vote).
3. `mqh/dead.py` — dead ball from: the 2nd-half pre-kickoff gap (frames 110808–113741), the ball
   outside the pitch, and a ball that stays still for 3 s. **These thresholds are untuned guesses.**
4. `mqh/decoder.py` — the possession decoder: a Viterbi with a switch penalty, so a team change
   needs sustained evidence, cheaper switches after a stoppage, and handovers placed inside silent
   gaps. `mqh/pipeline.py` wires everything together and
   picks up the optional stream modules `mqh/restarts.py` and `mqh/gaps.py` automatically
   (contracts in `docs/INTERFACES.md`; every constant and its reason in `docs/DECISIONS.md`).

## Rules we must not break

- No hand labels or hand corrections on this game, and no event data or other human-made data about it.
- No data in git (`data/` is ignored; so are caches and frame-level exports). Check `git status` before every push.
- The repo must be public by the deadline and must regenerate the submission file exactly.
- Do not upload the match data or extracts of it to any third-party hosted/cloud LLM (Terms, quoted in FAQ.md).

## Layout

```
make_submission.py   one command: data -> submissions/<team_name>.csv (+ validation)
mqh/                 our code (atd, dead, decode, decoder, pipeline; optional restarts, gaps)
tests/               decoder unit tests + synthetic-match test (no real data)
tools/               preflight, repro_check, diagnostics, sensitivity_synth
docs/                technical document draft, checklist, interfaces, decision log
evaluation/, games.py   organiser files, vendored unmodified (see THIRD_PARTY_NOTICES.md)
```
