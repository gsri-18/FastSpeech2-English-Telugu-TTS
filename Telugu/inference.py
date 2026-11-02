""" Synthesise the Telugu test sentences and plot the variance-adaptor outputs.

    python inference.py                       # latest checkpoint in checkpoints/
    python inference.py --restore_step 6000
    python inference.py --text "మీరు ఎలా ఉన్నారు?"   # any other sentence

Outputs go to ../Results/Telugu/ (see fastspeech2/inference_common.py).
The sentence text is kept exactly as given in the assignment, including the
split word in T3 ("అడ  విలో", with two spaces).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fastspeech2"))

from inference_common import run

TEST_SENTENCES = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--restore_step", type=int, default=None)
    parser.add_argument("--text", default=None, help="synthesise this text instead (saved as 'custom')")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    sentences = {"custom": args.text} if args.text else TEST_SENTENCES
    run(args.config, sentences, speaker_id=0, restore_step=args.restore_step,
        lang_name="Telugu",
        controls=(args.pitch_control, args.energy_control, args.duration_control),
        write_summary=not args.text)
