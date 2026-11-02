""" Plots of the variance adaptor's predictions for one synthesised utterance.

Three figures are saved per test sentence

  duration/<ID>.png  predicted duration per phoneme (ms), pauses in grey
  pitch/<ID>.png     predicted phoneme-level F0 (Hz) vs. F0 measured on the
                     synthesised waveform (pyworld), over the predicted mel
  energy/<ID>.png    predicted phoneme-level energy vs. frame energy of the
                     synthesised waveform, over the predicted mel
plus a JSON file with the underlying numbers.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw

SIL = {"sp", "spn", "sil"}


def measured_f0(wav_int16, sr, hop):
    x = wav_int16.astype(np.float64) / 32768.0
    f0, t = pw.dio(x, sr, frame_period=hop / sr * 1000)
    return pw.stonemask(x, f0, t, sr)


def measured_energy(wav_int16, stft):
    from audio.tools import get_mel_from_wav

    x = wav_int16.astype(np.float32) / 32768.0
    _, energy = get_mel_from_wav(x, stft)
    return energy


def _phone_axis(ax, phones, bounds, sec_per_frame):
    for b in bounds:
        ax.axvline(b * sec_per_frame, color="0.85", lw=0.5, zorder=0)
    for p, s, e in zip(phones, bounds[:-1], bounds[1:]):
        if e > s:
            ax.text((s + e) / 2 * sec_per_frame, 1.01, p, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6, rotation=90,
                    color="0.5" if p in SIL else "black")


def _step(values, bounds, sec_per_frame):
    xs, ys = [], []
    for v, s, e in zip(values, bounds[:-1], bounds[1:]):
        if e > s:
            xs += [s * sec_per_frame, e * sec_per_frame]
            ys += [v, v]
    return np.array(xs), np.array(ys)


def plot_all(uid, text, result, out_root, sr, hop, stft, lang_name):
    phones = result["phones"]
    dur = np.asarray(result["duration"])[: len(phones)]
    pitch = np.asarray(result["pitch_hz"])[: len(phones)]
    energy = np.asarray(result["energy"])[: len(phones)]
    mel = result["mel"]
    spf = hop / sr
    bounds = np.concatenate([[0], np.cumsum(dur)])
    t_mel = np.arange(mel.shape[1]) * spf
    f0 = measured_f0(result["wav"], sr, hop)[: mel.shape[1]]
    en = measured_energy(result["wav"], stft)[: mel.shape[1]]
    title = "{} ({}), {}".format(uid, lang_name, "{}")

    for sub in ("duration", "pitch", "energy"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    # duration
    fig, ax = plt.subplots(figsize=(max(8, 0.16 * len(phones)), 3.6))
    colors = ["0.7" if p in SIL else "tab:blue" for p in phones]
    ax.bar(np.arange(len(phones)), dur * spf * 1000, color=colors, width=0.8)
    ax.set_xticks(np.arange(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=6)
    ax.set_ylabel("predicted duration (ms)")
    ax.set_xlim(-0.6, len(phones) - 0.4)
    ax.set_title(title.format("duration predictor  (total {:.2f} s, {} phones)".format(
        bounds[-1] * spf, len(phones))), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "duration", uid + ".png"), dpi=160)
    plt.close(fig)

    # pitch and energy share a layout, with the mel on top and the contour below
    for kind, pred, meas, ylabel in [
        ("pitch", pitch, np.where(f0 > 0, f0, np.nan), "F0 (Hz)"),
        ("energy", energy, en, "energy (L2 norm of |STFT|)"),
    ]:
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(max(9, 0.14 * len(phones)), 5.2),
                                     sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
        a0.imshow(mel, origin="lower", aspect="auto", extent=[0, t_mel[-1] + spf, 0, mel.shape[0]])
        a0.set_ylabel("mel bin")
        a0.set_title(title.format("{} predictor".format(kind)), fontsize=9, pad=28)
        _phone_axis(a0, phones, bounds, spf)
        xs, ys = _step(pred, bounds, spf)
        a1.plot(xs, ys, color="tab:red", lw=1.8, label="predicted (phoneme level)")
        a1.plot(t_mel[: len(meas)], meas, color="0.35", lw=0.8,
                label="measured on synthesised audio")
        for b in bounds:
            a1.axvline(b * spf, color="0.9", lw=0.5, zorder=0)
        a1.set_ylabel(ylabel)
        a1.set_xlabel("time (s)")
        a1.legend(fontsize=7, loc="upper right")
        a1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, kind, uid + ".png"), dpi=160)
        plt.close(fig)

    voiced = f0[f0 > 0]
    data = {
        "id": uid,
        "text": text,
        "phones": phones,
        "duration_frames": dur.tolist(),
        "duration_ms": (dur * spf * 1000).round(1).tolist(),
        "pitch_hz": pitch.round(2).tolist(),
        "energy": energy.round(3).tolist(),
        "total_seconds": round(float(bounds[-1] * spf), 3),
        "measured_f0_mean_hz": round(float(voiced.mean()), 2) if len(voiced) else None,
        "measured_f0_std_hz": round(float(voiced.std()), 2) if len(voiced) else None,
        "voiced_fraction": round(float(len(voiced) / max(1, len(f0))), 3),
    }
    os.makedirs(os.path.join(out_root, "data"), exist_ok=True)
    with open(os.path.join(out_root, "data", uid + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data
