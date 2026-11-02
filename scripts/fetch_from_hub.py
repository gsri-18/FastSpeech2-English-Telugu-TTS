""" Download checkpoints + loss logs trained on Colab from the private HF repo.

    python scripts/fetch_from_hub.py --repo USER/REPO --lang English
    python scripts/fetch_from_hub.py --repo USER/REPO --lang Telugu --step 12000

Puts <step>.pth.tar into <Lang>/checkpoints/ (newest step unless --step) and
train_losses.csv / val_losses.csv into <Lang>/logs/.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))

from hub_sync import HubSync

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lang", required=True, choices=["English", "Telugu"])
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()

    lang_dir = os.path.join(ROOT, args.lang)
    sync = HubSync(args.repo, args.lang, os.path.join(lang_dir, "logs"), create=False)

    steps = sync.remote_steps()
    print("checkpoints on hub:", steps)
    step = args.step or max(steps)
    path = sync.pull(step, os.path.join(lang_dir, "checkpoints"))
    print("checkpoint ->", path)
    for name in ("train_losses.csv", "val_losses.csv"):
        p = os.path.join(sync.log_dir, name)
        if os.path.exists(p):
            print(name, "->", sum(1 for _ in open(p)) - 1, "rows")
