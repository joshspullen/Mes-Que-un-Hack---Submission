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
python tests/test_refine.py                                    # touch snapping unit tests, ~1 s
python tests/test_golden.py                                    # fixed made-up match must hash to tests/golden.json
python -m pytest tests/test_v4.py -q                           # engine v4: joint decoder, models, loaders
python tools/diagnostics.py submissions/<team_name>.csv        # per-half sanity numbers
python tools/stoppage_stats.py                                 # external-data stoppage rates (downloads Metrica samples)
cp submissions/<team_name>.csv evaluation/results/
python evaluation/run_evaluation.py --bootstrap 1000          # public 5 minutes: correctness only
```

The public 5 minutes (17 possessions) are for checking that the pipeline is wired correctly. **Do not
tune on them and do not train on them** (RULES.md).

## How it works (engine v4, the default)

1. `mqh/atd.py` reads the tracking onto the video timeline (`frame = frame_id - 1`).
2. `mqh/team_model.py` gives P(home has the ball) per frame: hand-built signals (nearest players,
   ball speed, team shape inside ball gaps from `mqh/gaps.py`, touches from `mqh/touches.py`) in a
   gradient-boosted model, antisymmetric by construction.
3. `mqh/dead_model.py` gives P(dead ball) per frame from 45 label-free tracking features.
4. `mqh/restarts.py` finds each Smart Tagging set-piece kick in the tracking; the tags are evidence
   for the stoppage and the restarting team, never labels.
5. `mqh/decoder3.py` is a joint home / away / dead Viterbi in integer arithmetic, so the file is
   identical on every machine. `mqh/v4.py` wires it together.

Both models are trained on **external matches only** (Metrica, SkillCorner, IDSSE/DFL; see
`docs/TECHNICAL_DOCUMENT.md`) and committed in `models/`; `tools/train_models.py --check` retrains
them and confirms the files are byte-identical. The earlier hand-built engine (v3: `mqh/pipeline.py`,
`mqh/decoder.py`) is kept and selectable with `python make_submission.py --engine v3`.
Every constant and the reason for it: `docs/DECISIONS.md`. Interfaces: `docs/INTERFACES.md`.

## Rules we must not break

- No hand labels or hand corrections on this game, and no event data or other human-made data about it.
- No data in git (`data/` is ignored; so are caches and frame-level exports). Check `git status` before every push.
- The repo must be public by the deadline and must regenerate the submission file exactly.
- Do not upload the match data or extracts of it to any third-party hosted/cloud LLM (Terms, quoted in FAQ.md).

## Layout

```
make_submission.py   one command: data -> submissions/<team_name>.csv (+ validation)
mqh/                 our code (engine v4: team_model, dead_model, restarts, decoder3, v4; v3 streams; external/)
models/              the two trained models + MANIFEST.json (external data only)
tests/               unit tests + synthetic-match test (no real data)
tools/               preflight, repro_check, diagnostics, external benchmarks, model training
docs/                technical document draft, checklist, interfaces, decision log
evaluation/, games.py   organiser files, vendored unmodified (see THIRD_PARTY_NOTICES.md)
```
