# Kaggle CI/CD scripts

Automates: push the notebook, watch the run, verify a valid submission was
produced, optionally submit it, and confirm Kaggle scored it.

## Why two monitors

- `kaggle_api_watch.py` uses only the official `kaggle` CLI (`kernels
  status`, `kernels output`). It is slower (polling) but its verdict is
  authoritative: it schema-checks the downloaded `submission.json` directly
  instead of guessing from log text. **Always trust this one.**
- `kaggle_live_monitor.py` uses DrissionPage to intercept an internal,
  undocumented Kaggle network endpoint via Chrome DevTools Protocol, for
  near-real-time log tailing so you don't wait out a full poll interval to
  see a crash. It is inherently fragile (Kaggle can change that endpoint's
  name or response shape without notice) and is visibility-only — `run_ci.py`
  never depends on it for pass/fail decisions.

`kaggle_submit_click.py` exists because Kaggle Code Competitions have **no
public API** for the "Submit to Competition" button — it's a website-only
action, so automating it requires the same CDP approach. It is off by
default (`--auto-submit` to enable) and always saves before/after
screenshots to `runs/kaggle_ci_screenshots/`.

## One-time setup

```bash
pip install kaggle DrissionPage
```

1. **Kaggle API token**: kaggle.com → Account → Create New API Token,
   downloads `kaggle.json`. Place it at `~/.kaggle/kaggle.json` (Windows:
   `C:\Users\<you>\.kaggle\kaggle.json`). Verify with `kaggle kernels list`.
2. **`kernel-metadata.json`** (repo root) is already wired to
   `ankitdash24/project-mythos-pipeline`, competition
   `arc-prize-2026-arc-agi-2`, `code_file` = the standalone notebook,
   `enable_internet: false` (Kaggle reruns are internet-disabled — matches
   the notebook's `AUTO_DOWNLOAD_*` defaults). Edit `enable_gpu` if you don't
   want GPU quota consumed for a fallback-mode smoke run.
3. **Chrome debug port** (only needed for `kaggle_live_monitor.py` /
   `--auto-submit`): close all Chrome windows first, then:
   ```
   chrome.exe --remote-debugging-port=9222
   ```
   Sign in to kaggle.com in that Chrome window before running anything.

## Usage

Watch-only, no Chrome needed (recommended default loop):

```bash
cd scripts/kaggle_ci
python run_ci.py --push --max-seconds 32400
```

Also live-tail the log in a second terminal while the above runs:

```bash
python kaggle_live_monitor.py
```

Full loop including the submit click (needs Chrome on :9222):

```bash
python run_ci.py --push --max-seconds 32400 --auto-submit
```

## When something goes wrong

`run_ci.py` stops at the first failing step and tells you which one. It
never silently retries or edits code. Look at:

- `runs/kaggle_ci_report.json` — fatal log lines + submission schema check
  from `kaggle_api_watch.py`.
- `runs/kaggle_ci_output/` — the actual files Kaggle produced (log,
  submission.json if any).
- `runs/kaggle_ci_screenshots/` — before/after the submit click, if
  `--auto-submit` was used.
- `runs/kaggle_score_report.json` — the raw submissions-API row from
  `check_submission_score.py`.

Bring any of those (or the kaggle.com error message) back and I'll help
diagnose it — this pipeline is built to surface errors clearly, not to
guess and resubmit on your behalf.

## Known limitations

- `kaggle_live_monitor.py`'s log-parsing is defensive but unverified against
  live Kaggle traffic (no Kaggle session was available while building this).
  If it reports "unrecognized packet shape," pass `--debug-dump raw.jsonl`,
  capture a few packets, and share them so the parser can be corrected.
- `kaggle_submit_click.py` finds the submit button by visible text
  (`page.ele("text:Submit to Competition")`), which is more resilient to
  Kaggle's CSS class churn than a raw selector but still breaks if Kaggle
  renames the button or changes the flow (e.g. a confirmation dialog).
  Screenshots are saved specifically so a broken click is obvious rather
  than silently wrong.
- Kaggle's actual GPU session time limit varies by competition; 9h
  (`--max-seconds 32400`) is a reasonable default but confirm the real cap
  for this competition's rules page and adjust.
