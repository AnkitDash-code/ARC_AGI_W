"""Click "Submit to Competition" on a completed kernel run via DrissionPage.

Kaggle Code Competitions have no public API for registering a submission
from a notebook run -- it's a website-only action. This script automates the
click; it is the most fragile piece of this whole pipeline because it
depends on Kaggle's current page text/DOM rather than any documented
contract. It deliberately does NOT run by default (see run_ci.py's
--auto-submit flag) and always saves a screenshot so you can see what the
page looked like if it didn't work.

Only run this after kaggle_api_watch.py has verified a schema-valid
submission.json was produced -- don't burn a daily submission slot on a
run that already failed.

Usage:
    python kaggle_submit_click.py --username ankitdash24 --slug project-mythos-pipeline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common import KAGGLE_USERNAME, KERNEL_SLUG

SUBMIT_BUTTON_TEXTS = ["Submit to Competition", "Submit to competition", "Submit"]


def click_submit(username: str, slug: str, debug_port: int, screenshot_dir: Path) -> int:
    try:
        from DrissionPage import ChromiumOptions, ChromiumPage
    except ImportError:
        print("DrissionPage is not installed. Run: pip install DrissionPage", file=sys.stderr)
        return 3

    co = ChromiumOptions()
    co.set_local_port(debug_port)
    try:
        page = ChromiumPage(co)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not connect to Chrome on port {debug_port}: {exc!r}", file=sys.stderr)
        return 3

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    kernel_url = f"https://www.kaggle.com/code/{username}/{slug}"
    print(f"Navigating to {kernel_url}")
    page.get(kernel_url)
    time.sleep(3)  # let the SPA finish rendering the latest version's action buttons

    before_shot = screenshot_dir / "before_submit_click.png"
    page.get_screenshot(path=str(before_shot))
    print(f"Saved {before_shot} -- check this if the click below fails or looks wrong.")

    button = None
    for text in SUBMIT_BUTTON_TEXTS:
        button = page.ele(f"text:{text}", timeout=5)
        if button:
            break

    if not button:
        print(
            "Could not find a 'Submit to Competition' button on the page. "
            "Kaggle's UI text/layout may differ from what this script expects, "
            f"or the latest version isn't finished committing yet. See {before_shot} "
            "and finish the submit manually on kaggle.com.",
            file=sys.stderr,
        )
        return 1

    button.click()
    time.sleep(3)
    after_shot = screenshot_dir / "after_submit_click.png"
    page.get_screenshot(path=str(after_shot))
    print(f"Clicked submit. Saved {after_shot} for confirmation.")
    print("Verify the result with check_submission_score.py once Kaggle finishes scoring.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=KAGGLE_USERNAME)
    parser.add_argument("--slug", default=KERNEL_SLUG)
    parser.add_argument("--debug-port", type=int, default=9222)
    parser.add_argument("--screenshot-dir", default="runs/kaggle_ci_screenshots")
    args = parser.parse_args(argv)
    return click_submit(args.username, args.slug, args.debug_port, Path(args.screenshot_dir))


if __name__ == "__main__":
    raise SystemExit(main())
