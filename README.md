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
- Binary detection is obtained by merging four-class predictions, **not by training a
  separate binary consensus objective**.
- The comparison-backbone adapter uses the same severity-head sequence as Dolphin:
  LayerNorm, Linear, GELU, Dropout, Linear. This applies to SenseVoice, wav2vec 2.0,
  and Whisper. LayerNorm was added after cleanup at the author's request; historical
  runs without it are a different configuration and must be rerun to report results
  for this version. Old head checkpoints are not directly compatible with this change.
  `coral_components.py` is used by this adapter and is not dead code.
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
system requirements, checkpoint downloads, and license. This cleanup did not download
or validate a real Dolphin checkpoint. Preserve the exact dependency versions from the
original training server for reproduction; the requirements file is not a tested lockfile.

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

Defaults: query dimension **32**, three clinician shifts, focal gamma **2.0**, alpha
strength **0.5**, batch size **8**, seed **43**, at most **100 epochs**, patience **20**,
AdamW learning rate **1e-4**, weight decay **1e-4**, CNN bottleneck **64**, and kernel **5**.
The clinician loss weight is 1.0. The consensus loss schedule is unchanged: epoch 1
uses 0.5, decreasing to 0.2 at epoch 11 and remaining there.

Path/environment overrides include `DATA_DIR`, `WAV_DIR`, `DOLPHIN_MODEL_DIR`,
`RESULTS_DIR`, `DEVICE`, `NUM_WORKERS`, `NUM_EPOCHS`, `BATCH_SIZE`, `RANDOM_SEEDS`,
`DOCTOR_QUERY_DIM`, `ENCODER_LAYER`, `FOCAL_GAMMA`, `FOCAL_USE_ALPHA`,
`FOCAL_ALPHA_STRENGTH`, `DOCTOR_LOSS_WEIGHT`, `MODE_LOSS_START_WEIGHT`,
`MODE_LOSS_END_WEIGHT`, and `MODE_LOSS_DECAY_EPOCHS`.
Use the defaults for the intended full-model configuration. Use a new output directory
for each run: existing output files can be overwritten.

Outputs are under `RESULTS_DIR/seed_43/fold_1` through `fold_5`, including per-fold
metrics, confusion matrices, training curves, clinician-shift diagnostics, and held-out
predictions. Root summaries distinguish fold means from pooled out-of-fold metrics.
As in the original script, the selected model is retained in memory for evaluation;
**checkpoints are not saved to disk**. `best_epochs.csv` records the selected epoch and
score only. Do not expect a checkpoint file from this release.

## Paper metrics, including merged binary detection

```bash
python summarize_results.py /path/to/new/output/seed_43
```

This dependency-free utility reads each fold's `outer_test_predictions.csv`, computes
metrics per fold, and prints mean and **sample standard deviation (ddof=1)** in percent.
Four-class F1, recall, and specificity are macro-averaged over labels 0 through 3.
Binary labels/predictions merge levels 1--3 into the positive class; binary F1 and recall
are positive-class metrics, not macro averages. Zero-denominator class metrics are 0.
The utility checks for duplicated patient IDs within/across test folds and never rewrites
manuscript values. Recomputed values, including standard deviations, depend on the supplied
prediction files; discrepancies with the paper must be reconciled from the experiment records.

## Historical backbone comparisons

Install `requirements-backbones.txt` in addition to the utility dependencies, and supply
a trusted, locally downloaded checkpoint. The adapter expects wav2vec 2.0 base (12 layers,
768 dimensions), Whisper small (12 layers, 768 dimensions), or SenseVoiceSmall (70 blocks,
512 dimensions). Exact checkpoint IDs/revisions must be recorded from the original runs.

```bash
ENCODER_BACKEND=whisper PRETRAINED_MODEL_DIR=/path/to/whisper-small \
DATA_DIR=/path/to/labels WAV_DIR=/path/to/wav RESULTS_DIR=/path/to/new/output \
python train.py
```

`ENCODER_BACKEND` also accepts `wav2vec2` and `sensevoice`. SenseVoice uses FunASR's
`trust_remote_code=True` behavior retained from the original adapter: use only code and
checkpoints you trust. See the architecture-change note above; the updated backbone
experiments have not yet been rerun or verified against the paper results.

## Checks

```bash
# No ML framework or patient data required:
python -B -m unittest -v test_metrics

# Requires PyTorch, TorchAudio, NumPy, pandas, scikit-learn and matplotlib;
# the tests substitute a tiny synthetic encoder, without downloading Dolphin:
python test_coral.py
python test_temporal_cnn_full_smoke.py
```

During cleanup, syntax/local-import checks and the synthetic metric tests passed.
The PyTorch-dependent tests and real-data training were **not run** because the local
test environment lacks those dependencies. These checks do not establish reproduction
of the paper results.

## Before publishing

Review staged files manually even with `.gitignore`; it does not remove already tracked
data. Never commit patient audio/labels/IDs, `.env` files, access tokens, cached checkpoints,
or result exports without the necessary permission. Keep the cleanup backup outside the
repository. Confirm the license, dependency versions, cohort mapping, original run config,
and backbone experiment versions before publishing. Add the actual GitHub URL to the manuscript only
after you create the repository; no repository has been published by this cleanup.
