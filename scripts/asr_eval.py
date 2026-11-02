""" Intelligibility check with Whisper, used to pick the checkpoints.

    python scripts/asr_eval.py checkpoints English              # every checkpoint found
    python scripts/asr_eval.py checkpoints Telugu 6000 7000
    python scripts/asr_eval.py tests English

The first form synthesises a fixed sample of validation sentences (every k-th
line of val.txt, 60 by default) from each checkpoint in its original speaker's
voice, transcribes them together with the real recordings (when raw_data/ is
present), and prints word and character error rates. The second form does the
same for the listed steps only, and the third transcribes the test sentences in
Results/<Lang>/test_generated_audios/.

Needs the checkpoints in <Lang>/checkpoints/ and `pip install transformers`.
English is transcribed with whisper-small.en and Telugu with whisper-large-v3-turbo.
"""

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))

import librosa
import torch
from scipy.io import wavfile
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from runtime import load_config
from synthesize import synthesize_utterance
from utils.model import get_model, get_vocoder
from utils.tools import device

WHISPER = {"English": ("openai/whisper-small.en", None),
           "Telugu": ("openai/whisper-large-v3-turbo", "telugu")}


class Transcriber:
    def __init__(self, lang):
        model, self.language = WHISPER[lang]
        self.asr = pipeline("automatic-speech-recognition", model=model, device=device,
                            torch_dtype=torch.float32)
        self.normalize = self.asr.tokenizer.normalize if self.language is None else BasicTextNormalizer()

    def __call__(self, wav_path):
        audio, _ = librosa.load(wav_path, sr=16000)
        kwargs = {"generate_kwargs": {"language": self.language, "task": "transcribe"}} if self.language else {}
        return self.asr(audio, **kwargs)["text"].strip()


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def error_rates(pairs, normalize):
    """WER and CER in percent over (reference, hypothesis) pairs."""
    we = wn = ce = cn = 0
    for ref, hyp in pairs:
        r, h = normalize(ref).split(), normalize(hyp).split()
        we += edit_distance(r, h)
        wn += len(r)
        ce += edit_distance(list("".join(r)), list("".join(h)))
        cn += len("".join(r))
    return 100 * we / wn, 100 * ce / cn, wn


def eval_checkpoints(lang, steps, n):
    configs = load_config(os.path.join(ROOT, lang, "config.yaml"))
    pre, model_cfg, _ = configs
    with open(os.path.join(pre["path"]["preprocessed_path"], "val.txt"), encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("|") for line in f]
    rows = rows[:: max(1, len(rows) // n)][:n]
    with open(os.path.join(pre["path"]["preprocessed_path"], "speakers.json")) as f:
        speaker_ids = json.load(f)
    refs = {}
    for name, spk, _, text in rows:
        raw = os.path.join(pre["path"]["raw_path"], spk, name + ".raw")
        refs[name] = open(raw, encoding="utf-8").read().strip() if os.path.exists(raw) else text

    vocoder = get_vocoder(model_cfg, device)
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    transcribe = Transcriber(lang)
    real = {name: os.path.join(pre["path"]["raw_path"], spk, name + ".wav") for name, spk, _, _ in rows}
    systems = {"real": real} if all(os.path.exists(p) for p in real.values()) else {}
    if not steps:
        steps = sorted({int(m.group(1)) for m in
                        (re.match(r"(\d+)(_inference)?\.pth\.tar$", f)
                         for f in os.listdir(os.path.join(ROOT, lang, "checkpoints"))) if m})
    with tempfile.TemporaryDirectory() as tmp:
        for step in steps:
            model = get_model(SimpleNamespace(restore_step=step), configs, device, train=False)
            out = systems[str(step)] = {}
            for name, spk, _, _ in rows:
                result = synthesize_utterance(model, configs, vocoder, refs[name], speaker_ids[spk])
                out[name] = os.path.join(tmp, "{}_{}.wav".format(step, name))
                wavfile.write(out[name], sr, result["wav"])
        print("{} sentences from val.txt".format(len(rows)))
        for system, wavs in systems.items():
            pairs = [(refs[name], transcribe(path)) for name, path in wavs.items()]
            wer, cer, words = error_rates(pairs, transcribe.normalize)
            print("{:>6}  WER {:5.1f}%  CER {:5.1f}%  ({} words)".format(system, wer, cer, words))


def eval_tests(lang):
    transcribe = Transcriber(lang)
    for path in sorted(glob.glob(os.path.join(ROOT, "Results", lang, "test_generated_audios", "*.wav"))):
        print("{:5s} {}".format(os.path.basename(path)[:-4], transcribe(path)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["checkpoints", "tests"])
    ap.add_argument("lang", choices=["English", "Telugu"])
    ap.add_argument("steps", type=int, nargs="*")
    ap.add_argument("--n", type=int, default=60, help="number of validation sentences")
    args = ap.parse_args()
    if args.mode == "checkpoints":
        eval_checkpoints(args.lang, args.steps, args.n)
    else:
        eval_tests(args.lang)
