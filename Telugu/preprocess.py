""" Telugu data preparation for FastSpeech 2.

    python preprocess.py config.yaml --stage prepare   # subset + resample + .lab
    python preprocess.py config.yaml --stage align     # G2P lexicon + MFA training/alignment
    python preprocess.py config.yaml --stage features  # mel / pitch / energy / duration
    python preprocess.py config.yaml                   # all three

Upstream had `prepare_align.py` (per-dataset text/audio dump), an external MFA
call with a pretrained model, and `preprocess.py`. There is no pretrained MFA
model for Telugu, so the `align` stage trains one on the selected subset using
the rule-based lexicon from fastspeech2/text/telugu.py.
"""

import argparse
import glob
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fastspeech2"))

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

from runtime import load_config
from text.telugu import word_to_phones, words_for_alignment
from preprocessor.preprocessor import Preprocessor

SPEAKER = "te_male"
_NON_TELUGU = re.compile(r"[A-Za-z0-9]")


def prepare(config):
    sub = config["subset"]
    paths = config["path"]
    sr_out = config["preprocessing"]["audio"]["sampling_rate"]
    shards = sorted(glob.glob(os.path.join(paths["corpus_path"], "*.parquet")))
    assert shards, "no parquet shards in {}".format(paths["corpus_path"])

    # First pass reads metadata only (duration from the wav header) and decodes nothing.
    candidates = []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["audio", "text", "gender"]).to_pylist()
            for i, row in enumerate(t):
                if row["gender"] != sub["gender"]:
                    continue
                if _NON_TELUGU.search(row["text"]):
                    continue  # stray Latin letters / digits in a handful of transcripts
                info = sf.info(io.BytesIO(row["audio"]["bytes"]))
                if not (sub["min_duration"] <= info.duration <= sub["max_duration"]):
                    continue
                name = os.path.splitext(os.path.basename(row["audio"]["path"]))[0]
                candidates.append((name, shard, rg, i, info.duration))
    total = sum(c[4] for c in candidates) / 3600
    print("{} candidate utterances, {:.2f} h".format(len(candidates), total))

    # Random subsample of train_hours + val_hours, then split. With one speaker a
    # plain random split is already stratified, and no utterance appears twice.
    rng = random.Random(sub["seed"])
    rng.shuffle(candidates)
    train, val, t_sec, v_sec = [], [], 0.0, 0.0
    for c in candidates:
        if v_sec < sub["val_hours"] * 3600:
            val.append(c)
            v_sec += c[4]
        elif t_sec < sub["train_hours"] * 3600:
            train.append(c)
            t_sec += c[4]
        else:
            break
    print("train {} utts {:.2f} h | val {} utts {:.2f} h".format(
        len(train), t_sec / 3600, len(val), v_sec / 3600))

    # Second pass decodes, resamples, peak-normalises and writes wav + .lab
    wanted = {(c[1], c[2], c[3]): c[0] for c in train + val}
    out_dir = os.path.join(paths["raw_path"], SPEAKER)
    os.makedirs(out_dir, exist_ok=True)
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in tqdm(range(pf.num_row_groups), desc=os.path.basename(shard)):
            keys = [k for k in wanted if k[0] == shard and k[1] == rg]
            if not keys:
                continue
            t = pf.read_row_group(rg, columns=["audio", "text"]).to_pylist()
            for _, _, i in keys:
                name = wanted[(shard, rg, i)]
                wav, sr = sf.read(io.BytesIO(t[i]["audio"]["bytes"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_out)
                # peak 0.95 of full scale (upstream used 32768, which overflows int16)
                wav = wav / max(1e-8, np.abs(wav).max()) * 0.95 * 32767
                wavfile.write(os.path.join(out_dir, name + ".wav"), sr_out, wav.astype(np.int16))
                with open(os.path.join(out_dir, name + ".lab"), "w", encoding="utf-8") as f:
                    f.write(" ".join(words_for_alignment(t[i]["text"])))
                with open(os.path.join(out_dir, name + ".raw"), "w", encoding="utf-8") as f:
                    f.write(t[i]["text"].strip())  # original transcript, for reference

    split = {"train": sorted(c[0] for c in train), "val": sorted(c[0] for c in val)}
    assert not set(split["train"]) & set(split["val"])
    with open(os.path.join(paths["raw_path"], "split.json"), "w") as f:
        json.dump(split, f, indent=1)
    with open(os.path.join(paths["raw_path"], "subset_stats.json"), "w") as f:
        json.dump({
            "candidates": len(candidates), "candidate_hours": total,
            "train_utts": len(train), "train_hours": t_sec / 3600,
            "val_utts": len(val), "val_hours": v_sec / 3600,
            "criteria": sub,
        }, f, indent=1)


def align(config, num_jobs, threading=False):
    paths = config["path"]
    os.makedirs(paths["mfa_path"], exist_ok=True)
    words = set()
    for lab in glob.glob(os.path.join(paths["raw_path"], "*", "*.lab")):
        with open(lab, encoding="utf-8") as f:
            words.update(f.read().split())
    dict_path = os.path.join(paths["mfa_path"], "telugu_lexicon.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        for w in sorted(words):
            ph = word_to_phones(w)
            if ph:
                f.write("{}\t{}\n".format(w, " ".join(ph)))
    print("lexicon: {} words -> {}".format(len(words), dict_path))

    mfa = shutil.which("mfa")
    assert mfa, "Montreal Forced Aligner not found, see requirements.txt for the conda command"
    tg_dir = os.path.join(paths["preprocessed_path"], "TextGrid")
    cmd = [
        mfa, "train", paths["raw_path"], dict_path,
        os.path.join(paths["mfa_path"], "telugu_mfa_acoustic.zip"),
        "--output_directory", tg_dir, "--clean", "--num_jobs", str(num_jobs),
        "--single_speaker", "--beam", "100", "--retry_beam", "400",
        # per-language MFA workspace (both corpora live in a dir called raw_data)
        "--temporary_directory", os.path.join(paths["mfa_path"], "tmp"),
    ]
    if threading:
        # fallback for when MFA's worker processes crash (much slower)
        cmd.append("--use_threading")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def features(config):
    with open(os.path.join(config["path"]["raw_path"], "split.json")) as f:
        split = json.load(f)
    Preprocessor(config, split=split).build_from_path()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--stage", choices=["prepare", "align", "features", "all"], default="all")
    parser.add_argument("--num_jobs", type=int, default=6)
    parser.add_argument("--mfa_threading", action="store_true")
    args = parser.parse_args()

    preprocess_config, _, _ = load_config(args.config)
    if args.stage in ("prepare", "all"):
        prepare(preprocess_config)
    if args.stage in ("align", "all"):
        align(preprocess_config, args.num_jobs, args.mfa_threading)
    if args.stage in ("features", "all"):
        features(preprocess_config)
