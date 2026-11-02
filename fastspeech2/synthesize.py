""" Text front-ends and single-utterance synthesis (upstream synthesize.py).

[MOD] Upstream's English front-end looked words up one by one in a lexicon,
its second front-end was Mandarin, and its CLI read three yaml files. Here the
front-ends are English (g2p_en on the whole sentence) and Telugu (rule-based
G2P), and synthesis is driven by English/inference.py and Telugu/inference.py.
"""

import json
import os
import re

import nltk
import numpy as np
import torch
from g2p_en import G2p

from utils.model import vocoder_infer
from utils.tools import device
from text import text_to_sequence, _clean_text
from text.numbers import _inflect
from text.telugu import text_to_phone_groups


def _time_words(hh, mm):
    hh, mm = int(hh), int(mm)
    words = _inflect.number_to_words(hh)
    if mm == 0:
        return words + " o'clock"
    if mm < 10:
        return words + " oh " + _inflect.number_to_words(mm)
    return words + " " + _inflect.number_to_words(mm)


def normalize_english(text):
    """Expand clock times and AM/PM, then apply the english_cleaners used for
    the training transcripts (ASCII, lower case, numbers, abbreviations).
    Upstream did no normalisation at inference, so "10:45" reached g2p_en as
    "ten", a pause, "forty-five"."""
    text = re.sub(
        r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\b\.?",
        lambda m: "{} {} em".format(
            _time_words(m.group(1), m.group(2)),
            "ay" if m.group(3).lower() == "a" else "pee",
        ),
        text,
    )
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: _time_words(m.group(1), m.group(2)), text)
    text = _clean_text(text, ["english_cleaners"])
    # inflect writes "twenty-six", and g2p_en would turn the hyphen into a pause
    return re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)


_g2p = None


def english_to_phones(text):
    """ARPAbet phones for a sentence, with "sp" at punctuation.

    g2p_en runs on the whole sentence, so it can POS-tag it and choose between
    heteronym pronunciations, e.g. "conduct" as a noun (K AA1 N D AH0 K T) or
    a verb (K AA0 N D AH1 K T). The MFA lexicon used for training was built
    from the same g2p_en / CMUdict, so the phone inventories match."""
    global _g2p
    if _g2p is None:
        # g2p_en fetches the tagger under its old name, newer NLTK looks for this one
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        _g2p = G2p()
    phones = []
    for p in _g2p(normalize_english(text)):
        if p == " ":
            continue
        if re.fullmatch(r"[^\w\s']+", p):  # punctuation
            if phones and phones[-1] != "sp":
                phones.append("sp")
            continue
        phones.append(p)
    while phones and phones[-1] == "sp":
        phones.pop()
    return phones


def telugu_to_phones(text):
    return [p for group in text_to_phone_groups(text) for p in group]


def text_to_phones(text, preprocess_config):
    language = preprocess_config["preprocessing"]["text"]["language"]
    if language == "en":
        return english_to_phones(text)
    if language == "te":
        return telugu_to_phones(text)
    raise ValueError("unsupported language: {}".format(language))


def preprocess_text(text, preprocess_config):
    """Convert text to its symbol id sequence and phone list."""
    phones = text_to_phones(text, preprocess_config)
    sequence = np.array(
        text_to_sequence(
            "{" + " ".join(phones) + "}",
            preprocess_config["preprocessing"]["text"]["text_cleaners"],
        )
    )
    assert len(sequence) == len(phones), "phone outside the symbol table: {}".format(phones)
    return sequence, phones


def synthesize_utterance(model, configs, vocoder, text, speaker_id=0, controls=(1.0, 1.0, 1.0)):
    """Synthesise one sentence and return the variance adaptor's predictions.

    Returns a dict with phones, wav (int16), mel, duration (frames per phone),
    log_duration, and phoneme-level pitch_hz / energy de-normalised with
    stats.json."""
    preprocess_config, model_config, _ = configs
    sequence, phones = preprocess_text(text, preprocess_config)
    speakers = torch.tensor([speaker_id], dtype=torch.long, device=device)
    texts = torch.from_numpy(sequence).long().unsqueeze(0).to(device)
    src_lens = torch.tensor([len(sequence)], device=device)
    p_control, e_control, d_control = controls
    with torch.no_grad():
        out = model(
            speakers, texts, src_lens, len(sequence),
            p_control=p_control, e_control=e_control, d_control=d_control,
        )
    mel_len = int(out[9][0].item())
    mel = out[1][0, :mel_len].transpose(0, 1)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
    p_mean, p_std = stats["pitch"][2:]
    e_mean, e_std = stats["energy"][2:]

    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav = vocoder_infer(mel.unsqueeze(0), vocoder, model_config, preprocess_config,
                        lengths=[mel_len * hop])[0]
    return {
        "phones": phones,
        "wav": wav,
        "mel": mel.cpu().numpy(),
        "duration": out[5][0].cpu().numpy().astype(int),
        "log_duration": out[4][0].cpu().numpy(),
        "pitch_hz": out[2][0].cpu().numpy() * p_std + p_mean,
        "energy": out[3][0].cpu().numpy() * e_std + e_mean,
    }
