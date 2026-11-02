""" Training / validation loss curves from <Lang>/logs/{train,val}_losses.csv.

    python scripts/plot_losses.py English
    python scripts/plot_losses.py Telugu

Writes Results/<Lang>/training_loss.png and validation_loss.png (total loss and
its five components) and train_vs_val_loss.png, the overlay used in the report.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPONENTS = ["mel", "mel_postnet", "pitch", "energy", "duration"]


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "time"}


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def curves(ax_total, ax_parts, d, k, label_prefix=""):
    s = d["step"]
    ax_total.plot(s, d["total"], color="tab:blue", alpha=0.25 if k > 1 else 1, lw=0.8)
    if k > 1:
        ax_total.plot(s[k - 1:], smooth(d["total"], k), color="tab:blue", lw=1.6,
                      label="{}total (moving avg, {} logs)".format(label_prefix, k))
    ax_parts_lines = []
    for name in COMPONENTS:
        y = smooth(d[name], k)
        ax_parts_lines += ax_parts.plot(s[len(s) - len(y):], y, lw=1.3, label=name)
    return ax_parts_lines


def main(lang):
    log_dir = os.path.join(ROOT, lang, "logs")
    out_dir = os.path.join(ROOT, "Results", lang)
    os.makedirs(out_dir, exist_ok=True)
    tr = read(os.path.join(log_dir, "train_losses.csv"))
    va = read(os.path.join(log_dir, "val_losses.csv"))

    for d, k, name, title in [(tr, 10, "training_loss.png", "Training loss"),
                              (va, 1, "validation_loss.png", "Validation loss")]:
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.2))
        curves(a0, a1, d, k)
        a0.set_title("{}, {} (total)".format(title, lang))
        a1.set_title("{}, {} (components)".format(title, lang))
        for a in (a0, a1):
            a.set_xlabel("step")
            a.set_ylabel("loss")
            a.grid(alpha=0.3)
        a1.set_yscale("log")
        a1.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))  # clear of the curves
        if k > 1:
            a0.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # smoothed training loss against validation loss
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ["total", "mel_postnet", "duration"]):
        k = 10
        ax.plot(tr["step"][k - 1:], smooth(tr[key], k), label="train (moving avg)")
        ax.plot(va["step"], va[key], "o-", ms=3, label="validation")
        ax.set_title("{}, {} loss".format(lang, key))
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_vs_val_loss.png"), dpi=150)
    plt.close(fig)

    best = int(np.argmin(va["total"]))
    print("{}: {} train logs, {} val points, best val total {:.4f} at step {}, last val {:.4f} at step {}".format(
        lang, len(tr["step"]), len(va["step"]), va["total"][best], int(va["step"][best]),
        va["total"][-1], int(va["step"][-1])))


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["English", "Telugu"]:
        main(lang)
