""" English (Svarah) data preparation for FastSpeech 2.

    python preprocess.py config.yaml --stage prepare   # subset + resample + .lab
    python preprocess.py config.yaml --stage align     # g2p_en lexicon + MFA english_us_arpa
    python preprocess.py config.yaml --stage features  # mel / pitch / energy / duration
    python preprocess.py config.yaml                   # all three

Svarah ships no speaker id. Its 8 demographic columns (gender, age group,
L1, state, district, qualification, job category, domain) give 115 distinct
combinations against the 117 speakers reported in the paper, so that tuple is
used as a pseudo speaker id for the multi-speaker model.
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
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fastspeech2"))

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

from runtime import load_config
from text import _clean_text
from preprocessor.preprocessor import Preprocessor

META_COLS = [
    "gender", "age-group", "primary_language", "native_place_state",
    "native_place_district", "highest_qualification", "job_category",
    "occupation_domain",
]
_ALLOWED = re.compile(r"^[A-Za-z0-9 ,.'?!\-:;&%$]+$")


def words_for_alignment(text, cleaners):
    """english_cleaners (ascii, lower-case, numbers/abbreviations expanded),
    then hyphens become spaces and all punctuation except apostrophes is removed."""
    text = _clean_text(text, cleaners)
    text = re.sub(r"[-–]", " ", text)
    text = re.sub(r"[^a-z' ]", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def prepare(config):
    sub = config["subset"]
    paths = config["path"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    sr_out = config["preprocessing"]["audio"]["sampling_rate"]
    shards = sorted(glob.glob(os.path.join(paths["corpus_path"], "*.parquet")))
    assert shards, "no parquet shards in {}".format(paths["corpus_path"])

    rows = []
    for shard in shards:
        t = pq.read_table(shard, columns=["duration", "text"] + META_COLS)
        for i, r in enumerate(t.to_pylist()):
            r["shard"], r["idx"] = shard, i
            rows.append(r)

    speaker_keys = sorted({"|".join(str(r[c]) for c in META_COLS) for r in rows})
    spk_of = {k: "spk{:03d}".format(i) for i, k in enumerate(speaker_keys)}

    candidates = []
    for r in rows:
        text = r["text"].strip()
        if not (sub["min_duration"] <= r["duration"] <= sub["max_duration"]):
            continue
        if not _ALLOWED.match(text):
            continue
        if re.search(r"\d{%d,}" % (sub["max_digit_run"] + 1), text):
            continue
        words = words_for_alignment(text, cleaners)
        if len(words) < sub["min_words"]:
            continue
        r["speaker"] = spk_of["|".join(str(r[c]) for c in META_COLS)]
        r["words"] = words
        candidates.append(r)
    total = sum(r["duration"] for r in candidates) / 3600
    print("{} candidate utterances, {:.2f} h, {} speakers".format(
        len(candidates), total, len({r["speaker"] for r in candidates})))

    # Split by speaker. Each speaker gives about val_ratio of its audio to
    # validation, so every validation speaker also has a trained embedding.
    # The rest goes to training, which is then randomly trimmed to train_hours.
    rng = random.Random(sub["seed"])
    val_ratio = sub["val_hours"] / total
    by_spk = defaultdict(list)
    for r in candidates:
        by_spk[r["speaker"]].append(r)
    train, val = [], []
    for spk in sorted(by_spk):
        utts = by_spk[spk]
        rng.shuffle(utts)
        target = val_ratio * sum(u["duration"] for u in utts)
        acc = 0.0
        for k, u in enumerate(utts):
            if acc < target and k < len(utts) - 1:
                val.append(u)
                acc += u["duration"]
            else:
                train.append(u)
    rng.shuffle(train)
    kept, t_sec = [], 0.0
    for u in train:
        if t_sec >= sub["train_hours"] * 3600:
            break
        kept.append(u)
        t_sec += u["duration"]
    train = kept
    v_sec = sum(u["duration"] for u in val)
    print("train {} utts {:.2f} h | val {} utts {:.2f} h".format(
        len(train), t_sec / 3600, len(val), v_sec / 3600))

    # Decode, resample from 16 to 22.05 kHz, peak-normalise and write wav + .lab
    by_shard = defaultdict(list)
    for u in train + val:
        by_shard[u["shard"]].append(u)
    for shard, utts in by_shard.items():
        audio = pq.read_table(shard, columns=["audio_filepath"]).column(0).to_pylist()
        for u in tqdm(utts, desc=os.path.basename(shard)):
            a = audio[u["idx"]]
            u["name"] = os.path.splitext(os.path.basename(a["path"]))[0]
            wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_out)
            wav = wav / max(1e-8, np.abs(wav).max()) * 0.95 * 32767
            out_dir = os.path.join(paths["raw_path"], u["speaker"])
            os.makedirs(out_dir, exist_ok=True)
            wavfile.write(os.path.join(out_dir, u["name"] + ".wav"), sr_out, wav.astype(np.int16))
            with open(os.path.join(out_dir, u["name"] + ".lab"), "w") as f:
                f.write(" ".join(u["words"]))
            with open(os.path.join(out_dir, u["name"] + ".raw"), "w") as f:
                f.write(u["text"].strip())

    split = {"train": sorted(u["name"] for u in train), "val": sorted(u["name"] for u in val)}
    assert not set(split["train"]) & set(split["val"])
    assert len(set(split["train"])) == len(train)
    with open(os.path.join(paths["raw_path"], "split.json"), "w") as f:
        json.dump(split, f, indent=1)
    with open(os.path.join(paths["raw_path"], "speaker_meta.json"), "w") as f:
        json.dump({v: dict(zip(META_COLS, k.split("|"))) for k, v in spk_of.items()}, f, indent=1)
    spk_hours = defaultdict(float)
    for u in train:
        spk_hours[u["speaker"]] += u["duration"] / 3600
    with open(os.path.join(paths["raw_path"], "subset_stats.json"), "w") as f:
        json.dump({
            "candidates": len(candidates), "candidate_hours": total,
            "train_utts": len(train), "train_hours": t_sec / 3600,
            "val_utts": len(val), "val_hours": v_sec / 3600,
            "train_speakers": len({u["speaker"] for u in train}),
            "val_speakers": len({u["speaker"] for u in val}),
            "train_minutes_per_speaker": {k: round(v * 60, 2) for k, v in sorted(spk_hours.items())},
            "criteria": sub,
        }, f, indent=1)


def build_lexicon(words, out_path):
    """Write a lexicon with the G2P used at inference (g2p_en). Words found in
    CMUdict get all their variants and MFA picks one per token. Other words get
    the g2p_en neural prediction."""
    from g2p_en import G2p

    g2p = G2p()
    with open(out_path, "w") as f:
        for w in sorted(words):
            prons = g2p.cmu.get(w) or [g2p.predict(w)]
            for p in prons:
                f.write("{}\t{}\n".format(w, " ".join(p)))


def align(config, num_jobs, threading=False):
    paths = config["path"]
    os.makedirs(paths["mfa_path"], exist_ok=True)
    words = set()
    for lab in glob.glob(os.path.join(paths["raw_path"], "*", "*.lab")):
        with open(lab) as f:
            words.update(f.read().split())
    dict_path = os.path.join(paths["mfa_path"], "english_g2p_en_lexicon.txt")
    build_lexicon(words, dict_path)
    print("lexicon: {} words -> {}".format(len(words), dict_path))

    mfa = shutil.which("mfa")
    assert mfa, "Montreal Forced Aligner not found, see requirements.txt for the conda command"
    subprocess.run([mfa, "model", "download", "acoustic", "english_us_arpa"], check=True)
    tg_dir = os.path.join(config["path"]["preprocessed_path"], "TextGrid")
    cmd = [mfa, "align", paths["raw_path"], dict_path, "english_us_arpa", tg_dir,
           "--clean", "--num_jobs", str(num_jobs), "--beam", "100", "--retry_beam", "400",
           # per-language MFA workspace (both corpora live in a dir called raw_data)
           "--temporary_directory", os.path.join(paths["mfa_path"], "tmp")]
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
