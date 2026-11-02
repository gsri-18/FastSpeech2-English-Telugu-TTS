""" Corpus statistics quoted in the report.

    python scripts/corpus_stats.py English
    python scripts/corpus_stats.py Telugu

Prints the pause (sp) durations and phone counts of the training set, read from
<Lang>/preprocessed_data/ (run preprocess.py first). For Telugu it also prints
the hours per speaker in the full IndicTTS export, measured from the WAV headers.
"""

import glob
import io
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))

from runtime import load_config


def training_set_stats(pre):
    ms_per_frame = pre["preprocessing"]["stft"]["hop_length"] / pre["preprocessing"]["audio"]["sampling_rate"] * 1000
    pauses, phones = [], Counter()
    with open(os.path.join(pre["path"]["preprocessed_path"], "train.txt"), encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("|") for line in f]
    for name, spk, seq, _ in rows:
        seq = seq.strip("{}").split()
        dur = np.load(os.path.join(pre["path"]["preprocessed_path"], "duration",
                                   "{}-duration-{}.npy".format(spk, name)))
        phones.update(p for p in seq if p != "sp")
        pauses += [d * ms_per_frame for p, d in zip(seq, dur) if p == "sp"]
    pauses = np.array(pauses)
    print("{} utterances, {} phones, {} pauses ({:.2f} per utterance)".format(
        len(rows), sum(phones.values()), len(pauses), len(pauses) / len(rows)))
    print("pause duration  median {:.0f} ms, mean {:.0f} ms, 90th percentile {:.0f} ms".format(
        np.median(pauses), pauses.mean(), np.percentile(pauses, 90)))
    print("rarest phones  " + ", ".join("{} {}".format(p, c) for p, c in sorted(phones.items(), key=lambda x: x[1])[:8]))


def telugu_speaker_hours(pre):
    import pyarrow.parquet as pq
    import soundfile as sf

    hours, utts, rates = defaultdict(float), Counter(), defaultdict(set)
    for shard in sorted(glob.glob(os.path.join(pre["path"]["corpus_path"], "*.parquet"))):
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg, columns=["audio", "gender"]).to_pylist():
                info = sf.info(io.BytesIO(row["audio"]["bytes"]))
                g = ["female", "male"][row["gender"]]
                hours[g] += info.duration / 3600
                utts[g] += 1
                rates[g].add(info.samplerate)
    for g in sorted(hours):
        print("{:6s} {:5d} utterances, {:5.2f} h, sample rate {}".format(g, utts[g], hours[g], sorted(rates[g])))


if __name__ == "__main__":
    lang = sys.argv[1]
    pre, _, _ = load_config(os.path.join(ROOT, lang, "config.yaml"))
    training_set_stats(pre)
    if lang == "Telugu":
        telugu_speaker_hours(pre)
