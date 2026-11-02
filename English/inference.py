""" Synthesise the English test sentences and plot the variance-adaptor outputs.

    python inference.py                         # latest checkpoint, default speaker
    python inference.py --speaker spk042        # any pseudo-speaker in speakers.json
    python inference.py --text "Hello there."   # any other sentence

The model is multi-speaker (Svarah pseudo-speakers). By default the speaker
with the most training audio is used (raw_data/subset_stats.json).
Outputs go to ../Results/English/ (see fastspeech2/inference_common.py).
The sentence text is kept exactly as given in the assignment, including the
split word in E3 ("rhyth  m", with two spaces).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fastspeech2"))

from inference_common import run
from runtime import load_config

TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth  m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}


def default_speaker(preprocess_config):
    with open(os.path.join(preprocess_config["path"]["raw_path"], "subset_stats.json")) as f:
        minutes = json.load(f)["train_minutes_per_speaker"]
    return max(minutes, key=minutes.get)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--restore_step", type=int, default=None)
    parser.add_argument("--speaker", default=None, help="pseudo-speaker name, e.g. spk042")
    parser.add_argument("--text", default=None, help="synthesise this text instead (saved as 'custom')")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    preprocess_config, _, _ = load_config(args.config)
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
        speaker_map = json.load(f)
    speaker = args.speaker or default_speaker(preprocess_config)
    print("Speaker: {} (id {})".format(speaker, speaker_map[speaker]))

    sentences = {"custom": args.text} if args.text else TEST_SENTENCES
    run(args.config, sentences, speaker_id=speaker_map[speaker],
        restore_step=args.restore_step, lang_name="English, " + speaker,
        controls=(args.pitch_control, args.energy_control, args.duration_control),
        write_summary=not args.text)
