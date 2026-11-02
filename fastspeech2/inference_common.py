""" Test-sentence synthesis shared by English/inference.py and Telugu/inference.py.

Everything is written under the language's result_path (Results/<Lang>/)

  test_generated_audios/<ID>.wav
  variance_adaptor_plots/{duration,pitch,energy}/<ID>.png
  variance_adaptor_plots/data/<ID>.json      (the plotted numbers)
  inference_summary.json

Sentences given with --text go to Results/<Lang>/custom/ instead, so the
submitted test outputs are never overwritten.
"""

import glob
import json
import os
import re
from types import SimpleNamespace

from scipy.io import wavfile

from runtime import load_config
from utils.model import get_model, get_vocoder
from utils.tools import device
from synthesize import synthesize_utterance
from utils.variance_plots import plot_all
import audio as Audio


def latest_step(ckpt_dir):
    steps = [int(m.group(1)) for m in
             (re.match(r"(\d+)(_inference)?\.pth\.tar$", os.path.basename(p))
              for p in glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))) if m]
    if not steps:
        raise FileNotFoundError("no checkpoints in {}".format(ckpt_dir))
    return max(steps)


def run(config_path, sentences, speaker_id=0, restore_step=None, lang_name="",
        controls=(1.0, 1.0, 1.0), write_summary=True):
    configs = load_config(config_path)
    preprocess_config, model_config, train_config = configs
    step = restore_step or latest_step(train_config["path"]["ckpt_path"])
    print("Loading checkpoint step {} on {}".format(step, device))

    model = get_model(SimpleNamespace(restore_step=step), configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    pp = preprocess_config["preprocessing"]
    sr, hop = pp["audio"]["sampling_rate"], pp["stft"]["hop_length"]
    stft = Audio.stft.TacotronSTFT(
        pp["stft"]["filter_length"], hop, pp["stft"]["win_length"],
        pp["mel"]["n_mel_channels"], sr, pp["mel"]["mel_fmin"], pp["mel"]["mel_fmax"],
    )

    out_root = train_config["path"]["result_path"]
    if not write_summary:
        out_root = os.path.join(out_root, "custom")
    wav_dir = os.path.join(out_root, "test_generated_audios")
    plot_dir = os.path.join(out_root, "variance_adaptor_plots")
    os.makedirs(wav_dir, exist_ok=True)

    summary = {"checkpoint_step": step, "speaker_id": speaker_id, "utterances": {}}
    for uid, text in sentences.items():
        result = synthesize_utterance(model, configs, vocoder, text, speaker_id, controls)
        wavfile.write(os.path.join(wav_dir, uid + ".wav"), sr, result["wav"])
        data = plot_all(uid, text, result, plot_dir, sr, hop, stft, lang_name)
        summary["utterances"][uid] = {
            "text": text, "phones": " ".join(result["phones"]),
            "seconds": data["total_seconds"],
            "measured_f0_mean_hz": data["measured_f0_mean_hz"],
        }
        print("{:4s} {:5.2f}s  {}".format(uid, data["total_seconds"], " ".join(result["phones"])))
    if write_summary:  # only for the test sentences, not for --text
        with open(os.path.join(out_root, "inference_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary
