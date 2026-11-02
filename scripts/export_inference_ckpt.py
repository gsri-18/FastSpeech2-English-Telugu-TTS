""" Shrink a training checkpoint to the weights needed for inference.

    python scripts/export_inference_ckpt.py English 9000

Reads <Lang>/checkpoints/<step>.pth.tar (model + optimizer state, 349 MB) and
writes <Lang>/checkpoints/<step>_inference.pth.tar with the model weights in
fp16 (56 MB, small enough for GitHub). inference.py loads it automatically
when the full checkpoint is absent.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    lang, step = sys.argv[1], int(sys.argv[2])
    ckpt_dir = os.path.join(ROOT, lang, "checkpoints")
    ckpt = torch.load(os.path.join(ckpt_dir, "{}.pth.tar".format(step)),
                      map_location="cpu", weights_only=False)
    model = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
    out = os.path.join(ckpt_dir, "{}_inference.pth.tar".format(step))
    torch.save({"model": model, "step": step}, out)
    print("{} ({:.1f} MB)".format(out, os.path.getsize(out) / 2**20))
