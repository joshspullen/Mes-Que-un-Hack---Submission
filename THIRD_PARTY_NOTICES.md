# Third-party notices

The following files are copied unmodified from the organisers' starter repository
(https://github.com/metrica-sports/more-than-a-hack-2027) and remain under its MIT licence:

- `games.py`
- `starter/load_smart_tagging.py` (Smart Tagging reader, used by `mqh/restarts.py` and
  `tools/inspect_set_pieces.py`)
- `evaluation/possession_metrics.py`, `evaluation/run_evaluation.py`, `evaluation/validate_submission.py`
- `evaluation/gt/BAR-ATH_possession_gt_public.csv` and `..._public_meta.json` (the public 5-minute
  ground truth; for checking the pipeline only, never for training or tuning)

`mqh/atd.py` follows the team-of-slot logic of `starter/baseline.py` from the same repository.

```
MIT License

Copyright (c) 2026-2027 Metrica Sports

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
