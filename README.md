# CAOL: Clinician-Aware Ordinal Learning

Research code for **Clinician-Aware Ordinal Learning for Speech-Based Dysarthria Severity Assessment**.
The default model is a frozen Dolphin-small encoder (layer 9, one-based), a single-scale
Temporal CNN, masked attention pooling, a shared severity score, ordered global thresholds,
and centered clinician-specific scalar shifts generated from learned queries.

## Scope and release status

- This directory contains code only: no clinical recordings, labels, trained checkpoints,
  or patient-level predictions are supplied or authorized for redistribution.
- The main workflow is five patient-independent outer folds, each with an 80:20
  training/validation split of the outer development set. Inner-validation macro-F1 selects
  the model; outer-test data are not used for selection.
- The original class-name mapping is retained in code (`normal`, `near_normal`,
  `mild_abnormal`, `abnormal`). Confirm its correspondence to the manuscript's clinical
  names; do not change numeric labels merely to rename categories.
- A software license has not been selected here. The authors should choose one before
  advertising the repository as open-source. Third-party models retain their own licenses.

## Installation

Use Python 3.10 or newer in a separate environment. Install matching PyTorch and
TorchAudio versions for your server using the [official installation instructions](https://pytorch.org/get-started/locally/).
Then install the Python utilities and Dolphin:

```bash
pip install -r requirements.txt
pip install dataoceanai-dolphin
```

The package name is `dataoceanai-dolphin`, **not** `dolphin`.
See the [upstream Dolphin repository](https://github.com/DataoceanAI/Dolphin) for its
system requirements, checkpoint downloads, and license. 

## Data format

Set `DATA_DIR` to a directory containing these nine original per-clinician tables:

```text
train0.xlsx  train1.xlsx  train2.xlsx
val0.xlsx    val1.xlsx    val2.xlsx
test0.xlsx   test1.xlsx   test2.xlsx
```

Each table must contain `patient_id` (or `ID`), `split`, and `speech_label`.
`wav_filename` is optional and defaults to `<patient_id>.wav`. The `split` values must
match the filename group: `train`, `val`, or `test`. Clinician indices 0, 1, 2 correspond
to the same clinicians across every table. Use anonymized string IDs and integer labels
0 through 3, with all three ratings present for the paper's CV workflow.

The three clinicians' tables must describe the same recordings within each group.
The original train/val/test groups are concatenated in that order and repartitioned by
five-fold stratification; their row order affects fold assignments. Retain original row
order and seed for reproduction. No patient may appear in more than one original group.
The existing loader reports and filters missing audio; check that report before relying
on cohort sizes or results. Audio is converted to mono at 16 kHz, with RMS normalization
for Dolphin. Dolphin's training-time SpecAugment remains enabled.

## Train the full model

Linux/server example (replace every example path):

```bash
DATA_DIR=/path/to/labels \
WAV_DIR=/path/to/wav \
DOLPHIN_MODEL_DIR=/path/to/dolphin/cache \
RESULTS_DIR=/path/to/new/output \
python train.py
```
Use the defaults for the intended full-model configuration. Use a new output directory
for each run: existing output files can be overwritten.
