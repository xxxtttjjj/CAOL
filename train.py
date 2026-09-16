import os
import random
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from dataset import PatientDataset, collate_fn, read_label_table
from evaluate import evaluate_model
from model import DolphinEncoderCORALClassifier as EncoderClassifier


NUM_CLASSES = 4
NUM_ANNOTATORS = 3
IGNORE_INDEX = -1

CLASS_NAMES = {
    0: "normal",
    1: "near_normal",
    2: "mild_abnormal",
    3: "abnormal",
}

LABEL_COLUMNS = [f"label{index}" for index in range(NUM_ANNOTATORS)]
REQUIRED_COLUMNS = ["patient_id", "split", "wav_filename", *LABEL_COLUMNS]

ENCODER_BACKEND = "dolphin"

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "labels"
WAV_DIR = PROJECT_DIR / "data" / "wav"
RESULTS_DIR = PROJECT_DIR / "outputs" / ENCODER_BACKEND

DEVICE = "cuda:0"


DOCTOR_LOSS_WEIGHT = 1.0
MODE_LOSS_START_WEIGHT = 0.5
MODE_LOSS_END_WEIGHT = 0.2
MODE_LOSS_DECAY_EPOCHS = 10

FOCAL_GAMMA = 2.0
FOCAL_USE_ALPHA = 1
FOCAL_ALPHA_STRENGTH = 0.5


ENCODER_LAYER = 9

TEMPORAL_CNN_BOTTLENECK_DIM = 64
TEMPORAL_CNN_KERNEL_SIZE = 5
TEMPORAL_CNN_DROPOUT = 0.1

CORAL_MIN_THRESHOLD_GAP = 0.1

DOCTOR_QUERY_DIM = 32
DOCTOR_NUM_ATTENTION_HEADS = 2
DOCTOR_NUM_SELF_ATTENTION_LAYERS = 1
DOCTOR_QUERY_DROPOUT = 0.1
DOCTOR_FFN_MULTIPLIER = 2


BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 2

RANDOM_SEEDS = [43]
CV_NUM_FOLDS = 5
INNER_VAL_RATIO = 0.20
EARLY_STOPPING_PATIENCE = 20

LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE = 5e-7
WEIGHT_DECAY = 1e-4

USE_SPECAUG = True

SPLIT_LABEL_FILES = {
    split: [
        DATA_DIR / f"{split}{index}.xlsx"
        for index in range(NUM_ANNOTATORS)
    ]
    for split in ("train", "val", "test")
}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def _model_kwargs(device: str) -> dict:
    return {
        "dolphin_model_name": "small",
        "dolphin_model_dir": (
            os.environ.get("DOLPHIN_MODEL_DIR", "").strip() or None
        ),
        "device": device,
        "use_layer": ENCODER_LAYER,
        "freeze_s2t": True,
        "freeze_layer": 12,
        "use_specaug": USE_SPECAUG,
        "head_hidden_size": 64,
        "head_dropout_rate": 0.5,
        "num_classes": NUM_CLASSES,
        "temporal_cnn_bottleneck_dim": TEMPORAL_CNN_BOTTLENECK_DIM,
        "temporal_cnn_kernel_size": TEMPORAL_CNN_KERNEL_SIZE,
        "temporal_cnn_dropout_rate": TEMPORAL_CNN_DROPOUT,
        "attention_hidden_dim": 32,
        "attention_dropout_rate": 0.1,
        "num_doctors": NUM_ANNOTATORS,
        "coral_min_threshold_gap": CORAL_MIN_THRESHOLD_GAP,
        "doctor_query_dim": DOCTOR_QUERY_DIM,
        "doctor_num_attention_heads": DOCTOR_NUM_ATTENTION_HEADS,
        "doctor_num_self_attention_layers": DOCTOR_NUM_SELF_ATTENTION_LAYERS,
        "doctor_query_dropout": DOCTOR_QUERY_DROPOUT,
        "doctor_ffn_multiplier": DOCTOR_FFN_MULTIPLIER,
    }


def build_model(device: str):
    return EncoderClassifier(**_model_kwargs(device)).to(device)


def build_loader(
    dataset: PatientDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )



def normalize_single_label_df(
    path: Path,
    label_column: str,
) -> pd.DataFrame:
    df = read_label_table(path)

    if "patient_id" not in df.columns and "ID" in df.columns:
        df = df.rename(columns={"ID": "patient_id"})

    required = {"patient_id", "split", "speech_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    if "wav_filename" not in df.columns:
        df["wav_filename"] = df["patient_id"] + ".wav"

    df["wav_filename"] = df["wav_filename"].astype(str).str.strip()

    df[label_column] = (
        pd.to_numeric(df["speech_label"], errors="coerce")
        .fillna(IGNORE_INDEX)
        .astype(int)
    )

    valid_labels = {IGNORE_INDEX, *range(NUM_CLASSES)}
    invalid = sorted(set(df[label_column]) - valid_labels)
    if invalid:
        raise ValueError(f"Invalid labels in {path}: {invalid}")

    if df["patient_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate patient IDs.")

    return df[["patient_id", "split", "wav_filename", label_column]]


def load_split_df(split_name: str) -> pd.DataFrame:
    tables = []

    for index, path in enumerate(SPLIT_LABEL_FILES[split_name]):
        table = normalize_single_label_df(path, f"label{index}")

        if set(table["split"]) != {split_name}:
            raise ValueError(f"{path} contains an unexpected split value.")

        tables.append(table)

    merged = tables[0]

    for index, table in enumerate(tables[1:], start=1):
        merged = merged.merge(
            table[["patient_id", "wav_filename", f"label{index}"]],
            on=["patient_id", "wav_filename"],
            how="inner",
            validate="one_to_one",
        )

    return merged[REQUIRED_COLUMNS].reset_index(drop=True)


def load_fixed_splits() -> Dict[str, pd.DataFrame]:
    return {
        split: load_split_df(split)
        for split in SPLIT_LABEL_FILES
    }


def median_consensus_labels(df: pd.DataFrame) -> np.ndarray:
    consensus = []

    for row in df[LABEL_COLUMNS].to_numpy(dtype=int):
        valid = row[row >= 0]
        consensus.append(int(np.median(valid)))

    return np.asarray(consensus, dtype=int)


def median_consensus_from_batch(
    hard_labels: torch.Tensor,
) -> torch.Tensor:
    medians = []

    for row in hard_labels.long():
        valid = row[row >= 0]

        if valid.numel() == 0:
            raise ValueError(
                "Each sample needs at least one valid doctor label."
            )

        ordered = valid.sort().values
        middle = ordered.numel() // 2

        medians.append(
            ordered[middle]
            if ordered.numel() % 2
            else (ordered[middle - 1] + ordered[middle]) // 2
        )

    return torch.stack(medians)


def dataset_from_df(df: pd.DataFrame) -> PatientDataset:
    return PatientDataset(
        df[REQUIRED_COLUMNS].copy(),
        WAV_DIR,
        normalize=ENCODER_BACKEND == "dolphin",
        num_classes=NUM_CLASSES,
        label_columns=LABEL_COLUMNS,
    )


def format_split_counts(df: pd.DataFrame) -> str:
    labels = median_consensus_labels(df)
    counts = np.bincount(labels, minlength=NUM_CLASSES)

    return ", ".join(
        f"{CLASS_NAMES[index]}={count}"
        for index, count in enumerate(counts)
    )



def labels_to_coral_targets(
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    levels = torch.arange(
        num_classes - 1,
        device=labels.device,
    )
    return (labels.unsqueeze(-1) > levels).float()


def get_mode_loss_weight(
    epoch: int,
    start_weight: float,
    end_weight: float,
    decay_epochs: int,
) -> float:
    progress = min(max(epoch, 0) / decay_epochs, 1.0)
    return float(
        start_weight
        + progress * (end_weight - start_weight)
    )


def focal_binary_cross_entropy_with_logits(
    logits,
    targets,
    gamma,
    alpha_pos=None,
    reduction="mean",
):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    weight = (1 - torch.exp(-bce)) ** gamma

    if alpha_pos is not None:
        alpha_pos = alpha_pos.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        weight = weight * (
            alpha_pos * targets
            + (1 - alpha_pos) * (1 - targets)
        )

    loss = weight * bce

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()


def blend_focal_alpha(
    raw_alpha_pos: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if not 0 <= strength <= 1:
        raise ValueError(
            "Focal alpha strength must be in [0, 1]."
        )

    return 0.5 + float(strength) * (
        raw_alpha_pos.float() - 0.5
    )


class MultiAnnotatorCORALLoss(torch.nn.Module):
    """Focal CORAL loss for doctor labels and their median consensus."""

    def __init__(
        self,
        num_classes: int,
        doctor_loss_weight: float,
        mode_loss_start_weight: float,
        mode_loss_end_weight: float,
        mode_loss_decay_epochs: int,
        doctor_shift_reg_weight: float = 0.0,
        ignore_index: int = -1,
        focal_gamma: float = 2.0,
        focal_alpha_enabled: bool = False,
        doctor_alpha_pos: torch.Tensor = None,
        global_alpha_pos: torch.Tensor = None,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.doctor_loss_weight = float(doctor_loss_weight)
        self.mode_loss_start_weight = float(mode_loss_start_weight)
        self.mode_loss_end_weight = float(mode_loss_end_weight)
        self.mode_loss_decay_epochs = int(mode_loss_decay_epochs)
        self.ignore_index = int(ignore_index)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha_enabled = bool(focal_alpha_enabled)
        self.current_epoch = 0

        self.register_buffer(
            "doctor_alpha_pos",
            (
                torch.empty(0)
                if doctor_alpha_pos is None
                else doctor_alpha_pos.detach().float()
            ),
        )

        self.register_buffer(
            "global_alpha_pos",
            (
                torch.empty(0)
                if global_alpha_pos is None
                else global_alpha_pos.detach().float()
            ),
        )

    def set_epoch(self, epoch: int):
        self.current_epoch = max(int(epoch) - 1, 0)

    @property
    def mode_loss_weight(self) -> float:
        return get_mode_loss_weight(
            self.current_epoch,
            self.mode_loss_start_weight,
            self.mode_loss_end_weight,
            self.mode_loss_decay_epochs,
        )

    def loss_components(
        self,
        outputs,
        soft_targets=None,
        hard_labels=None,
    ):
        del soft_targets

        doctor_labels = hard_labels.long()
        doctor_logits = outputs["doctor_coral_logits"]

        valid_mask = doctor_labels != self.ignore_index
        safe_labels = doctor_labels.masked_fill(~valid_mask, 0)

        doctor_targets = labels_to_coral_targets(
            safe_labels,
            self.num_classes,
        )

        doctor_losses = focal_binary_cross_entropy_with_logits(
            doctor_logits,
            doctor_targets,
            self.focal_gamma,
            (
                self.doctor_alpha_pos
                if self.focal_alpha_enabled
                else None
            ),
            reduction="none",
        ).mean(dim=-1)

        doctor_loss = (
            self.doctor_loss_weight
            * doctor_losses[valid_mask].mean()
        )

        consensus_labels = median_consensus_from_batch(
            doctor_labels
        )
        consensus_targets = labels_to_coral_targets(
            consensus_labels,
            self.num_classes,
        )

        consensus_loss = self.mode_loss_weight * (
            focal_binary_cross_entropy_with_logits(
                outputs["global_coral_logits"],
                consensus_targets,
                self.focal_gamma,
                (
                    self.global_alpha_pos
                    if self.focal_alpha_enabled
                    else None
                ),
            )
        )

        return {
            "total_loss": doctor_loss + consensus_loss,
            "doctor_coral_loss": doctor_loss,
            "consensus_loss": consensus_loss,
            "mode_loss_weight": self.mode_loss_weight,
        }

    def forward(
        self,
        outputs,
        soft_targets=None,
        hard_labels=None,
    ):
        return self.loss_components(
            outputs,
            soft_targets,
            hard_labels,
        )["total_loss"]


def build_criterion(train_dataset: PatientDataset):
    doctor_labels = torch.tensor(
        train_dataset.df[LABEL_COLUMNS].to_numpy(),
        dtype=torch.long,
    )

    valid = doctor_labels != IGNORE_INDEX
    safe_labels = doctor_labels.masked_fill(~valid, 0)

    doctor_targets = labels_to_coral_targets(
        safe_labels,
        NUM_CLASSES,
    )

    valid_float = valid.unsqueeze(-1).float()

    positive_rates = (
        (doctor_targets * valid_float).sum(dim=0)
        / valid_float.sum(dim=0).clamp_min(1)
    )

    consensus_labels = torch.tensor(
        median_consensus_labels(train_dataset.df),
        dtype=torch.long,
    )

    consensus_targets = labels_to_coral_targets(
        consensus_labels,
        NUM_CLASSES,
    )

    doctor_alpha = blend_focal_alpha(
        1 - positive_rates,
        FOCAL_ALPHA_STRENGTH,
    )

    global_alpha = blend_focal_alpha(
        1 - consensus_targets.mean(dim=0),
        FOCAL_ALPHA_STRENGTH,
    )

    return MultiAnnotatorCORALLoss(
        num_classes=NUM_CLASSES,
        doctor_loss_weight=DOCTOR_LOSS_WEIGHT,
        mode_loss_start_weight=MODE_LOSS_START_WEIGHT,
        mode_loss_end_weight=MODE_LOSS_END_WEIGHT,
        mode_loss_decay_epochs=MODE_LOSS_DECAY_EPOCHS,
        ignore_index=IGNORE_INDEX,
        focal_gamma=FOCAL_GAMMA,
        focal_alpha_enabled=FOCAL_USE_ALPHA,
        doctor_alpha_pos=doctor_alpha,
        global_alpha_pos=global_alpha,
    )




def save_missing_audio_report(
    datasets: Dict[str, PatientDataset],
    output_dir: Path,
):
    reports = []

    for split_name, dataset in datasets.items():
        missing = getattr(dataset, "missing_df", pd.DataFrame())

        if not missing.empty:
            missing = missing.copy()
            missing.insert(0, "dataset_split", split_name)
            reports.append(missing)

    if reports:
        report = pd.concat(reports, ignore_index=True)

        report.to_csv(
            output_dir / "missing_audio_files.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Missing audio files: {len(report)}")


def build_optimizer(
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    head_parameters = [
        *model.temporal_cnn.parameters(),
        *model.attention_pooling.parameters(),
        *model.severity_head.parameters(),
        *model.doctor_shift_module.parameters(),
        model.threshold_start,
    ]

    seen = {
        id(parameter)
        for parameter in head_parameters
    }

    groups = [
        {
            "params": [
                parameter
                for parameter in head_parameters
                if parameter.requires_grad
            ],
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        }
    ]

    encoder_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
        and id(parameter) not in seen
    ]

    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": ENCODER_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        )

    return torch.optim.AdamW(groups)


@torch.no_grad()
def evaluate_split(
    model,
    dataset,
    criterion,
) -> Dict[str, object]:
    loader = build_loader(
        dataset,
        BATCH_SIZE,
        shuffle=False,
    )

    return evaluate_model(
        model,
        loader,
        criterion,
        DEVICE,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
    )


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
) -> dict:
    model.train()

    totals = {
        "total_loss": 0.0,
        "doctor_coral_loss": 0.0,
        "consensus_loss": 0.0,
    }
    total_samples = 0

    for (
        wavs,
        lengths,
        _soft_labels,
        hard_labels,
        _patient_ids,
    ) in train_loader:
        wavs = wavs.to(
            DEVICE,
            non_blocking=True,
        )
        lengths = lengths.to(
            DEVICE,
            non_blocking=True,
        )
        hard_labels = hard_labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        components = criterion.loss_components(
            model(wavs, lengths),
            hard_labels=hard_labels,
        )

        loss = components["total_loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss encountered."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            5.0,
        )

        optimizer.step()

        batch_size = hard_labels.size(0)

        for name in totals:
            totals[name] += (
                float(components[name].item())
                * batch_size
            )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Cannot train with an empty data loader."
        )

    return {
        name: value / total_samples
        for name, value in totals.items()
    }




def metric_row(
    prefix: str,
    evaluation: Dict[str, object],
) -> Dict[str, float]:
    row = {
        f"{prefix}_loss": float(
            evaluation["total_loss"]
        ),
        f"{prefix}_doctor_coral_loss": float(
            evaluation["doctor_coral_loss"]
        ),
        f"{prefix}_consensus_loss": float(
            evaluation["consensus_loss"]
        ),
    }

    metric_sets = {
        "median": evaluation["consensus_metrics"]
    }

    for label_name in LABEL_COLUMNS:
        metric_sets[label_name] = evaluation[
            f"{label_name}_metrics"
        ]
        metric_sets[f"{label_name}_head"] = evaluation[
            f"{label_name}_head_metrics"
        ]

    for label_name, metrics in metric_sets.items():
        for metric in (
            "acc",
            "balanced_acc",
            "macro_f1",
            "weighted_f1",
            "qwk",
            "mae",
        ):
            row[
                f"{prefix}_{label_name}_{metric}"
            ] = float(metrics[metric])

    return row


def save_metric_details(
    output_dir: Path,
    split_name: str,
    evaluation: Dict[str, object],
):
    metric_sets = {
        "median": evaluation["consensus_metrics"]
    }

    for label_name in LABEL_COLUMNS:
        metric_sets[label_name] = evaluation[
            f"{label_name}_metrics"
        ]
        metric_sets[f"{label_name}_head"] = evaluation[
            f"{label_name}_head_metrics"
        ]

    per_class_rows = []

    for scope, metrics in metric_sets.items():
        pd.DataFrame(
            metrics["confusion_matrix"]
        ).to_csv(
            output_dir
            / f"{split_name}_confusion_matrix_{scope}.csv",
            index=False,
        )

        for class_id, values in metrics["per_class"].items():
            per_class_rows.append(
                {
                    "scope": scope,
                    "class_id": class_id,
                    **values,
                }
            )

    pd.DataFrame(per_class_rows).to_csv(
        output_dir
        / f"{split_name}_per_class_metrics.csv",
        index=False,
    )


def save_prediction_table(
    output_dir: Path,
    evaluation: Dict[str, object],
    outer_fold: int,
):
    rows = {
        "patient_id": evaluation["patient_ids"],
        "outer_fold": outer_fold,
        "true_median": evaluation["consensus_labels"],
        "pred": evaluation["preds"],
        "true_binary": evaluation[
            "binary_consensus_labels"
        ],
        "binary_pred": evaluation[
            "binary_predictions"
        ],
        "prob_abnormal": evaluation[
            "binary_probs"
        ][:, 1],
    }

    for class_id in range(NUM_CLASSES):
        rows[f"global_prob_{class_id}"] = (
            evaluation["probs"][:, class_id]
        )

    for doctor_index, label_name in enumerate(
        LABEL_COLUMNS
    ):
        rows[label_name] = evaluation[
            "hard_labels"
        ][:, doctor_index]

        rows[f"doctor{doctor_index}_pred"] = (
            evaluation["annotator_preds"][
                :, doctor_index
            ]
        )

    pd.DataFrame(rows).to_csv(
        output_dir / "outer_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_training_curves(
    history: pd.DataFrame,
    output_dir: Path,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        label="Train",
    )
    axes[0].plot(
        history["epoch"],
        history["val_loss"],
        label="Validation",
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Loss",
        title="Loss",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        history["epoch"],
        history["val_macro_f1"],
        label="Macro-F1",
    )
    axes[1].plot(
        history["epoch"],
        history["val_qwk"],
        label="QWK",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Score",
        title="Validation metrics",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "training_curves.png",
        dpi=180,
    )
    plt.close(fig)


def aggregate_outer_oof_metrics(
    oof_df: pd.DataFrame,
) -> Dict[str, object]:
    true_labels = oof_df[
        "true_median"
    ].to_numpy(dtype=int)

    predictions = oof_df[
        "pred"
    ].to_numpy(dtype=int)

    class_ids = list(range(NUM_CLASSES))

    precision, recall, per_class_f1, support = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=class_ids,
            average=None,
            zero_division=0,
        )
    )

    probability_columns = [
        f"global_prob_{index}"
        for index in class_ids
    ]

    try:
        auc = float(
            roc_auc_score(
                true_labels,
                oof_df[
                    probability_columns
                ].to_numpy(),
                labels=class_ids,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = float("nan")

    return {
        "acc": float(
            accuracy_score(
                true_labels,
                predictions,
            )
        ),
        "balanced_acc": float(
            balanced_accuracy_score(
                true_labels,
                predictions,
            )
        ),
        "macro_precision": float(
            np.mean(precision)
        ),
        "macro_recall": float(
            np.mean(recall)
        ),
        "macro_f1": float(
            np.mean(per_class_f1)
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "auc_macro_ovr": auc,
        "qwk": float(
            cohen_kappa_score(
                true_labels,
                predictions,
                weights="quadratic",
            )
        ),
        "mae": float(
            np.mean(
                np.abs(
                    true_labels - predictions
                )
            )
        ),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predictions,
            labels=class_ids,
        ),
        "per_class": pd.DataFrame(
            {
                "class_id": class_ids,
                "class_name": [
                    CLASS_NAMES[index]
                    for index in class_ids
                ],
                "precision": precision,
                "recall": recall,
                "f1": per_class_f1,
                "support": support,
            }
        ),
    }



def run_one_cv_fold(
    seed: int,
    fold_index: int,
    outer_train_df: pd.DataFrame,
    outer_test_df: pd.DataFrame,
    root_output_dir: Path,
) -> Dict[str, object]:
    set_seed(seed + fold_index)

    output_dir = (
        root_output_dir
        / f"fold_{fold_index}"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=INNER_VAL_RATIO,
        random_state=seed * 100 + fold_index,
    )

    labels = median_consensus_labels(
        outer_train_df
    )

    train_indices, val_indices = next(
        splitter.split(
            outer_train_df,
            labels,
        )
    )

    train_df = outer_train_df.iloc[
        train_indices
    ].reset_index(drop=True)

    val_df = outer_train_df.iloc[
        val_indices
    ].reset_index(drop=True)

    print(
        f"\n========== Fold "
        f"{fold_index}/{CV_NUM_FOLDS} =========="
    )

    for name, frame in (
        ("train", train_df),
        ("validation", val_df),
        ("test", outer_test_df),
    ):
        print(
            f"{name}: n={len(frame)} | "
            f"{format_split_counts(frame)}"
        )

    train_dataset = dataset_from_df(train_df)
    val_dataset = dataset_from_df(val_df)
    test_dataset = dataset_from_df(outer_test_df)

    save_missing_audio_report(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        },
        output_dir,
    )

    train_loader = build_loader(
        train_dataset,
        BATCH_SIZE,
        shuffle=True,
    )

    model = build_model(DEVICE)
    criterion = build_criterion(
        train_dataset
    ).to(DEVICE)
    optimizer = build_optimizer(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Parameters: {total_parameters:,} total, "
        f"{trainable_parameters:,} trainable"
    )

    best_score = -float("inf")
    best_epoch = 0
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, NUM_EPOCHS + 1):
        criterion.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_metrics = evaluate_split(
            model,
            val_dataset,
            criterion,
        )

        val_summary = val_metrics[
            "consensus_metrics"
        ]
        score = float(
            val_summary["macro_f1"]
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics[
                    "total_loss"
                ],
                "train_doctor_coral_loss":
                    train_metrics[
                        "doctor_coral_loss"
                    ],
                "train_consensus_loss":
                    train_metrics[
                        "consensus_loss"
                    ],
                "val_loss": val_metrics[
                    "total_loss"
                ],
                "val_doctor_coral_loss":
                    val_metrics[
                        "doctor_coral_loss"
                    ],
                "val_consensus_loss":
                    val_metrics[
                        "consensus_loss"
                    ],
                "mode_loss_weight":
                    val_metrics[
                        "mode_loss_weight"
                    ],
                "val_macro_f1": score,
                "val_qwk": float(
                    val_summary["qwk"]
                ),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['total_loss']:.5f} | "
            f"val={val_metrics['total_loss']:.5f} | "
            f"macro_f1={score:.4f} | "
            f"qwk={val_summary['qwk']:.4f}"
        )

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_loss = float(
                val_metrics["total_loss"]
            )

            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"Early stopping; "
                f"best epoch: {best_epoch}"
            )
            break

    model.load_state_dict(best_state)
    model.to(DEVICE)

    criterion.set_epoch(best_epoch)

    val_metrics = evaluate_split(
        model,
        val_dataset,
        criterion,
    )

    test_metrics = evaluate_split(
        model,
        test_dataset,
        criterion,
    )

    history = pd.DataFrame(history_rows)

    history.to_csv(
        output_dir / "loss_history.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "criterion": "val_macro_f1",
                "best_epoch": best_epoch,
                "best_value": best_score,
            }
        ]
    ).to_csv(
        output_dir / "best_epochs.csv",
        index=False,
    )

    plot_training_curves(
        history,
        output_dir,
    )

    save_metric_details(
        output_dir,
        "inner_val",
        val_metrics,
    )

    save_metric_details(
        output_dir,
        "outer_test",
        test_metrics,
    )

    save_prediction_table(
        output_dir,
        test_metrics,
        fold_index,
    )

    row = {
        "seed": seed,
        "outer_fold": fold_index,
        "inner_train_size": len(train_dataset),
        "inner_val_size": len(val_dataset),
        "outer_test_size": len(test_dataset),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "best_val_loss": best_val_loss,
        **metric_row(
            "inner_val",
            val_metrics,
        ),
        **metric_row(
            "outer_test",
            test_metrics,
        ),
    }

    pd.DataFrame([row]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    summary = test_metrics[
        "consensus_metrics"
    ]

    print(
        f"Fold {fold_index} test | "
        f"accuracy={summary['acc']:.4f} | "
        f"macro_f1={summary['macro_f1']:.4f} | "
        f"qwk={summary['qwk']:.4f}"
    )

    return row


def run_nested_five_fold_cv(
    seed: int,
    root_output_dir: Path,
) -> Dict[str, object]:
    set_seed(seed)

    root_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_dfs = load_fixed_splits()

    all_df = pd.concat(
        split_dfs.values(),
        ignore_index=True,
    )

    consensus = median_consensus_labels(
        all_df
    )

    print(
        f"Nested five-fold CV | "
        f"seed={seed} | "
        f"n={len(all_df)} | "
        f"backend={ENCODER_BACKEND} | "
        f"device={DEVICE}"
    )

    splitter = StratifiedKFold(
        n_splits=CV_NUM_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    fold_rows = []
    oof_frames = []
    assignments = []

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            all_df,
            consensus,
        ),
        start=1,
    ):
        fold_rows.append(
            run_one_cv_fold(
                seed,
                fold_index,
                all_df.iloc[
                    train_indices
                ].reset_index(drop=True),
                all_df.iloc[
                    test_indices
                ].reset_index(drop=True),
                root_output_dir,
            )
        )

        oof_frames.append(
            pd.read_csv(
                root_output_dir
                / f"fold_{fold_index}"
                / "outer_test_predictions.csv",
                dtype={"patient_id": str},
            )
        )

        assignments.append(
            pd.DataFrame(
                {
                    "outer_fold": fold_index,
                    "patient_id":
                        all_df.iloc[
                            test_indices
                        ]["patient_id"].to_numpy(),
                    "true_median":
                        consensus[test_indices],
                }
            )
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.concat(
        assignments,
        ignore_index=True,
    ).to_csv(
        root_output_dir
        / "outer_fold_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_results = pd.DataFrame(
        fold_rows
    )

    fold_results.to_csv(
        root_output_dir
        / "outer_fold_metrics.csv",
        index=False,
    )

    summary_rows = []

    summary_columns = [
        column
        for column in fold_results
        if (
            column == "best_epoch"
            or column.startswith(
                "outer_test_median_"
            )
        )
    ]

    for column in summary_columns:
        values = fold_results[
            column
        ].astype(float)

        summary_rows.append(
            {
                "metric": column,
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std(ddof=1)
                ),
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        root_output_dir
        / "outer_fold_summary.csv",
        index=False,
    )

    oof_df = (
        pd.concat(
            oof_frames,
            ignore_index=True,
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    oof_df.to_csv(
        root_output_dir
        / "outer_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled = aggregate_outer_oof_metrics(
        oof_df
    )

    pd.DataFrame(
        pooled["confusion_matrix"]
    ).to_csv(
        root_output_dir
        / "pooled_oof_confusion_matrix_median.csv",
        index=False,
    )

    pooled["per_class"].to_csv(
        root_output_dir
        / "pooled_oof_per_class.csv",
        index=False,
    )

    scalar_metrics = {
        f"outer_oof_{name}": value
        for name, value in pooled.items()
        if name not in {
            "confusion_matrix",
            "per_class",
        }
    }

    pd.DataFrame(
        [
            {
                "seed": seed,
                **scalar_metrics,
            }
        ]
    ).to_csv(
        root_output_dir
        / "pooled_oof_metrics.csv",
        index=False,
    )

    print(
        f"Pooled OOF | "
        f"accuracy={pooled['acc']:.4f} | "
        f"macro_f1={pooled['macro_f1']:.4f} | "
        f"qwk={pooled['qwk']:.4f}"
    )

    return scalar_metrics


def run_five_fold_cv_experiment():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_rows = []

    for seed in RANDOM_SEEDS:
        metrics = run_nested_five_fold_cv(
            seed,
            RESULTS_DIR / f"seed_{seed}",
        )

        seed_rows.append(
            {
                "seed": seed,
                **metrics,
            }
        )

    results = pd.DataFrame(seed_rows)

    results.to_csv(
        RESULTS_DIR
        / "nested_oof_results_by_seed.csv",
        index=False,
    )

    if len(results) > 1:
        rows = []

        for column in results:
            if column.startswith("outer_oof_"):
                values = results[
                    column
                ].astype(float)

                rows.append(
                    {
                        "metric": column,
                        "mean": float(
                            values.mean()
                        ),
                        "std": float(
                            values.std(ddof=1)
                        ),
                    }
                )

        pd.DataFrame(rows).to_csv(
            RESULTS_DIR
            / "nested_oof_summary_by_seed.csv",
            index=False,
        )


def run_with_training_log():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        RESULTS_DIR
        / f"training_{timestamp}.log"
    )

    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        with (
            redirect_stdout(
                TeeStream(
                    sys.stdout,
                    log_file,
                )
            ),
            redirect_stderr(
                TeeStream(
                    sys.stderr,
                    log_file,
                )
            ),
        ):
            print(
                f"Python: "
                f"{sys.version.split()[0]} | "
                f"PyTorch: {torch.__version__}"
            )

            run_five_fold_cv_experiment()

    print(
        f"Training log saved to: "
        f"{log_path}"
    )


if __name__ == "__main__":
    run_with_training_log()
