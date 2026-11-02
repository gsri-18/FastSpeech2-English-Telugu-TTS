# FastSpeech 2 for English and Telugu

Two separate [FastSpeech 2](https://arxiv.org/abs/2006.04558) text-to-speech models built on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2).

| | English | Telugu |
|---|---|---|
| Data | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English, 115 pseudo-speakers) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (one male speaker) |
| Subset | 5.00 h train (3,451 utts), 1.11 h val (774 utts) | 5.00 h train (2,625 utts), 1.00 h val (531 utts) |
| Text front-end | g2p_en (CMUdict and POS-based heteronyms), ARPAbet | rule-based Telugu G2P written for this project |
| Alignment | MFA 3, pretrained `english_us_arpa` | MFA 3, acoustic model trained on the subset |
| Model | 4 encoder and 4 decoder FFT blocks, 29.4 M parameters | same, 29.3 M parameters (no speaker embedding) |
| Training | 12k steps, batch 48, bf16, Colab A100 (24 min) | 12k steps, batch 48, bf16, Colab A100 (24 min) |
| Lowest validation loss at a saved checkpoint | 2.509 (step 8k) | 2.106 (step 7k) |
| Checkpoint used | step 9,000 | step 6,000 |
| Vocoder | HiFi-GAN universal (from upstream) | same |

The full write-up is in [Report/Technical_Report.pdf](Report/Technical_Report.pdf).

## Links

| | |
|---|---|
| Code, results, report and inference checkpoints | [github.com/gsri-18/FastSpeech2-English-Telugu-TTS](https://github.com/gsri-18/FastSpeech2-English-Telugu-TTS) |
| Full training checkpoints and TensorBoard logs | [huggingface.co/gsri-18/FastSpeech2-English-Telugu-TTS](https://huggingface.co/gsri-18/FastSpeech2-English-Telugu-TTS) |

Everything needed to run the trained models is in this repository. The Colab notebooks read their inputs from a
private dataset repo (`gsri-18/fs2-iiith-data`, the code and preprocessed features) and wrote every checkpoint
to a private working repo (`gsri-18/fastspeech2-en-te-checkpoints`). The selected checkpoints and the logs from
that working repo are published on the Hugging Face page above.

## Repository layout

```
English/, Telugu/            one folder per language
  config.yaml                preprocess, model and train settings (replaces upstream's 3 yaml files)
  preprocess.py              data selection, MFA alignment, mel/pitch/energy/duration features
  train.py                   training entry point (local or Colab)
  inference.py               synthesises the test sentences and plots the variance adaptor
  checkpoints/               <step>_inference.pth.tar  (fp16 weights, 56 MB)
  raw_data/                  split.json (train/val lists), subset_stats.json, speaker_meta.json (English)
  preprocessed_data/         train.txt, val.txt, stats.json, speakers.json
  mfa/                       pronunciation lexicon (and the trained Telugu aligner)
  logs/                      train_losses.csv, val_losses.csv (with IST timestamps), and for Telugu
                             colab_preprocess_output.txt (the log of the Colab preprocessing run)
Results/<Lang>/
  training_loss.png, validation_loss.png, train_vs_val_loss.png
  test_generated_audios/     <ID>.wav for the assignment's test sentences
  variance_adaptor_plots/    duration/, pitch/, energy/ (and data/ with the plotted numbers)
  checkpoint_selection.json  validation loss and ASR intelligibility used to pick the checkpoint
  inference_summary.json     text, phones and length of every test sentence
fastspeech2/                 the model code (upstream, changes marked "# [MOD]")
  config_upstream/           upstream's LJSpeech yaml files, kept for reference
  text/telugu.py             Telugu number normalisation and G2P
  hub_sync.py                checkpoint and log mirroring to a private Hugging Face repo (Colab)
  utils/variance_plots.py    duration, pitch and energy plots
scripts/                     loss plots, checkpoint export, ASR evaluation, corpus statistics,
                             Colab data bundling and fetching
colab/                       the notebooks that trained both models (with their outputs) and
                             the Telugu preprocessing notebook
Report/Technical_Report.pdf  (and its LaTeX source)
Report/listening_checks/     Whisper and Gemini transcripts and judgements of the test sentences,
                             with the exact prompts
requirements.txt
```

Changes to upstream files are marked with `# [MOD]`, so `grep -rn "\[MOD\]" fastspeech2` finds them. The first
commit is upstream's code, unmodified apart from its configs moving to `config_upstream/`, and `git diff --ignore-cr-at-eol <first commit> -- fastspeech2` shows
the complete record (upstream's `train.py`, `evaluate.py` and `synthesize.py` had Windows line endings).

## Setup

```bash
conda create -n fs2 -c conda-forge python=3.11 montreal-forced-aligner=3 ffmpeg
conda activate fs2
pip install -r requirements.txt
```

MFA is only needed for preprocessing. Inference needs just the Python packages, and the HiFi-GAN vocoder
(`fastspeech2/hifigan/generator_universal.pth.tar`) is included. The first English synthesis downloads a
few small NLTK resources that g2p_en uses.

## Synthesis with the trained models

```bash
(cd English && python inference.py)    # writes E1 to E4 to Results/English/
(cd Telugu && python inference.py)     # writes T1 to T4 to Results/Telugu/
```

Other options are `--text "any sentence"` (written to `Results/<Lang>/custom/`), `--restore_step N`, `--pitch_control`, `--energy_control`,
`--duration_control` and, for English, `--speaker spkNNN`. The default English speaker is spk073, who has the
most training audio. I generated all results on a MacBook (Apple MPS), and the same commands run unchanged on a
CUDA GPU.

## Reproducing training

1. **Data.** Svarah is gated, so accept its terms on Hugging Face and run `hf auth login` first.
   ```bash
   hf download ai4bharat/Svarah --repo-type dataset --local-dir downloads/svarah_hf
   ln -s svarah_hf/data downloads/Svarah            # the *.parquet shards
   hf download SPRINGLab/IndicTTS_Telugu --repo-type dataset --revision refs/convert/parquet \
       --include "default/train/*" --local-dir downloads/telugu_hf
   ln -s telugu_hf/default/train downloads/IndicTTS_Telugu
   ```
2. **Preprocess.** This is CPU work. Telugu alignment took 2.8 h on 8 cores because an aligner is trained.
   ```bash
   (cd English && python preprocess.py)   # or --stage prepare | align | features
   (cd Telugu && python preprocess.py)
   ```
   The subsets are deterministic (seed 1234) and reproduce the committed `raw_data/split.json` files.
3. **Train.** A GPU is recommended.
   ```bash
   (cd English && python train.py)        # --restore_step N to resume
   ```
   Both models were trained on a Colab A100 with [colab/train_English.ipynb](colab/train_English.ipynb) and
   [colab/train_Telugu.ipynb](colab/train_Telugu.ipynb), which keep every output and cell finish time. They pull
   the preprocessed features from a private Hugging Face dataset repo and push checkpoints and logs to a
   private model repo at every checkpoint (`--resume --hub_repo USER/REPO`). Telugu preprocessing ran on a Colab
   CPU runtime. [colab/preprocess_Telugu.ipynb](colab/preprocess_Telugu.ipynb) runs the same steps, and the log of
   that run is `Telugu/logs/colab_preprocess_output.txt`.
4. **Plots and inference weights.**
   ```bash
   python scripts/plot_losses.py English
   python scripts/export_inference_ckpt.py English 9000    # full training checkpoint to 56 MB of fp16 weights
   ```

## Evaluation and statistics

The checkpoint comparison and the corpus numbers in the report come from two scripts.

```bash
pip install transformers                                  # only needed for asr_eval.py
python scripts/asr_eval.py checkpoints English            # every checkpoint in English/checkpoints
python scripts/asr_eval.py tests Telugu                   # transcribes Results/Telugu/test_generated_audios
python scripts/corpus_stats.py Telugu                     # pause lengths, rare phones, hours per speaker
```

The full training checkpoints (English 9000, Telugu 6000) can be downloaded from the Hugging Face page into
`<Lang>/checkpoints/`.

## Acknowledgements

- FastSpeech 2 implementation from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT), including its HiFi-GAN vocoder.
- Svarah by AI4Bharat. Javed et al., *Svarah: Evaluating English ASR Systems on Indian Accents*, Interspeech 2023.
- Indic TTS Database by IIT Madras, Hugging Face export by SPRINGLab.

The transcript lists in `<Lang>/preprocessed_data/` come from these datasets and remain under their terms.
- Montreal Forced Aligner, g2p_en and pyworld.
