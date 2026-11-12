# Submission checklist

**Deadline: Tuesday 17 November 2026.** Aim to submit by midday so there is a buffer. The last form
response before the deadline is the one scored; there is no live leaderboard and no feedback.

Owners: **I** = integrator, **B** = dead ball, **C** = decoder, **D** = report. Fill in names.

## Schedule

| date | what | owner |
|---|---|---|
| Wed 11 Nov | everyone clones, runs `python tests/test_pipeline.py` and the real-data command, claims a task | all |
| Thu 12 – Fri 13 | first pull requests (dead ball, decoder); report sections that don't need final numbers | B, C, D |
| Sat 14 | merge, run on real data, read possession count / durations / `D` share | I |
| Sun 15 | **method freeze**: from here only bug fixes | all |
| Mon 16 | final run, `repro_check`, `preflight --final`, report finished, repo made public | I, D |
| Tue 17 (morning) | submit the form; open it once afterwards to check the upload | I |

## Rules that disqualify (everyone)

- [ ] No hand labels or hand corrections on this game: not by eye, not in a spreadsheet, not at transitions only.
- [ ] No tuning, fitting or calibrating on this game's ground truth or video. Parameters are either reasoned (written down) or tuned on *external* data we have the rights to.
- [ ] No human-made data about this game (event data, corrected tracking, annotations).
- [ ] No match data, extracts, or video uploaded to any hosted AI service or third-party cloud LLM. Run everything locally.
- [ ] The file is produced by code in this repo and the repo regenerates it exactly.
- [ ] The public 5 minutes are for checking the pipeline is wired correctly, never for choosing between versions.

## Before every commit

- [ ] `git status` lists nothing from `data/`, `.venv/`, caches, `*.log` or `submissions/*.csv`.
- [ ] Never `git push --force`. Work on a branch; open a pull request.

## Final run (Mon 16 Nov)

- [ ] Method frozen; `git pull` on `main`; everything merged.
- [ ] Decide the real team name (exactly as in the registration form): **TEAM_NAME**
- [ ] `pip freeze > requirements-lock.txt`, commit it.
- [ ] `python make_submission.py --team-name TEAM_NAME`
- [ ] Sanity: possession count and mean duration plausible; `D` share plausible; both halves look similar; `X` only outside the graded halves.
- [ ] `python tools/repro_check.py` prints PASS; paste the sha256 into the technical document.
- [ ] Fresh clone proof: clone to a new folder, copy data in, install, run `repro_check`, same hash. Write down who and when in the report.
- [ ] `python tools/preflight.py --team-name TEAM_NAME --final` shows 0 FAIL.
- [ ] Report: no TODOs left; saved/exported as `TEAM_NAME.pdf` or `TEAM_NAME.md`.
- [ ] `git add -f submissions/TEAM_NAME.csv` (deliberate; optional), commit, push.

## Make the repo public (after the checks above)

- [ ] On GitHub, browse `data/`: only `.gitkeep`. Search the repo for `.mp4`, `_atd`, `homography`: none.
- [ ] Repo Settings, then General, then Danger Zone, then Change visibility, then Public.
- [ ] Open the repo in a private/incognito window to confirm it loads without logging in.

## Submit

- [ ] The form link arrives by email to registered participants; you need a Google account.
- [ ] Fields: team name, `TEAM_NAME.csv` (about 1.9 MB), public repo URL, `TEAM_NAME.pdf|md`.
- [ ] Run `python evaluation/validate_submission.py` on the **exact file you upload**.
- [ ] One response per team; you may edit it until the deadline.
