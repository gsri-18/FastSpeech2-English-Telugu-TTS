""" Train the Telugu FastSpeech 2 model.

    python train.py                          # from scratch, settings in config.yaml
    python train.py --restore_step 8000      # resume from checkpoints/8000.pth.tar
    python train.py --resume --hub_repo USER/REPO --num_workers 4
                                             # on Colab, resume from the newest local or Hub
                                             # checkpoint and upload each new one

Training/validation losses are written to logs/train_losses.csv and
logs/val_losses.csv (plus TensorBoard event files in logs/train, logs/val).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANG = os.path.basename(HERE)
# fastspeech2/ goes first on sys.path because its train.py holds the training
# loop. This file is only the per-language entry point.
sys.path.insert(0, os.path.join(HERE, "..", "fastspeech2"))

from runtime import load_config
import hub_sync
import train as fs2_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="restore the newest checkpoint (local or --hub_repo)")
    parser.add_argument("--hub_repo", default=None,
                        help="private HF model repo to mirror checkpoints/logs to")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", default=None,
                        choices=["auto", "fp32", "bf16", "fp16"], help="override config")
    parser.add_argument("--quiet", action="store_true", help="hide the per-epoch progress bar")
    args = parser.parse_args()

    configs = load_config(args.config)
    sync = hub_sync.setup(args, configs[2], LANG)
    fs2_train.main(args, configs)
    if sync is not None:
        sync.upload_logs("end")  # includes the final validation row
        sync.wait()
