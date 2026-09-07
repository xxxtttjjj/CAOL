from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


CLASS_NAMES = {
    0: "normal",
    1: "near_normal",
    2: "mild_abnormal",
    3: "abnormal",
}


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
    class_names: Dict[int, str],
) -> Dict[str, object]:
    class_ids = list(range(num_classes))
    precision, recall, per_class_f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=class_ids,
            average=None,
            zero_division=0,
        )
    )

    try:
        auc = float(
            roc_auc_score(
                labels,
                probabilities,
                labels=class_ids,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = float("nan")

    try:
        qwk = float(cohen_kappa_score(labels, predictions, weights="quadratic"))
    except ValueError:
        qwk = float("nan")

    return {
        "acc": float(accuracy_score(labels, predictions)),
        "balanced_acc": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(per_class_f1)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "auc_macro_ovr": auc,
        "qwk": qwk,
        "mae": float(np.mean(np.abs(labels - predictions))),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=class_ids
        ),
        "per_class": {
            class_id: {
                "name": class_names.get(class_id, f"class_{class_id}"),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(per_class_f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in class_ids
        },
    }


def _median_consensus(hard_labels: np.ndarray) -> np.ndarray:
    consensus = []
    for row in hard_labels:
        valid_labels = row[row >= 0]
        if valid_labels.size == 0:
            raise ValueError(
                "Every evaluated sample needs at least one valid doctor label."
            )
        consensus.append(int(np.median(valid_labels)))
    return np.asarray(consensus, dtype=int)


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    criterion,
    device,
    num_classes: int = 4,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """Evaluate global and doctor-specific CORAL predictions."""
    model.eval()
    class_names = class_names or CLASS_NAMES
    loss_names = ("total_loss", "doctor_coral_loss", "consensus_loss")
    loss_totals = {name: 0.0 for name in loss_names}
    mode_loss_weight = 0.0
    total_samples = 0

    collected = {
        "probs": [],
        "preds": [],
        "doctor_probs": [],
        "doctor_preds": [],
        "hard_labels": [],
    }
    patient_ids = []
    global_thresholds = doctor_shifts = doctor_thresholds = None

    for wavs, lengths, _soft_labels, hard_labels, batch_patient_ids in data_loader:
        wavs = wavs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        hard_labels_device = hard_labels.to(device, non_blocking=True)

        outputs = model(wavs, lengths)
        components = criterion.loss_components(
            outputs, hard_labels=hard_labels_device
        )
        batch_size = hard_labels.size(0)
        for name in loss_names:
            loss_totals[name] += float(components[name].item()) * batch_size
        mode_loss_weight = float(components["mode_loss_weight"])
        total_samples += batch_size

        collected["probs"].append(outputs["global_class_probs"].cpu())
        collected["preds"].append(outputs["global_predictions"].cpu())
        collected["doctor_probs"].append(outputs["doctor_class_probs"].cpu())
        collected["doctor_preds"].append(outputs["doctor_predictions"].cpu())
        collected["hard_labels"].append(hard_labels.cpu())
        patient_ids.extend(str(value) for value in batch_patient_ids)

        global_thresholds = outputs["global_thresholds"].cpu()
        doctor_shifts = outputs["doctor_shifts"].cpu()
        doctor_thresholds = outputs["doctor_thresholds"].cpu()

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    probabilities = torch.cat(collected["probs"]).numpy().astype(np.float64)
    predictions = torch.cat(collected["preds"]).numpy().astype(int)
    doctor_probabilities = (
        torch.cat(collected["doctor_probs"]).numpy().astype(np.float64)
    )
    doctor_predictions = torch.cat(collected["doctor_preds"]).numpy().astype(int)
    hard_labels = torch.cat(collected["hard_labels"]).numpy().astype(int)
    consensus_labels = _median_consensus(hard_labels)

    consensus_metrics = _classification_metrics(
        consensus_labels,
        predictions,
        probabilities,
        num_classes,
        class_names,
    )
    annotator_metrics = {}
    annotator_head_metrics = {}
    for doctor_index in range(hard_labels.shape[1]):
        valid = hard_labels[:, doctor_index] >= 0
        label_name = f"label{doctor_index}"
        annotator_metrics[label_name] = _classification_metrics(
            hard_labels[valid, doctor_index],
            predictions[valid],
            probabilities[valid],
            num_classes,
            class_names,
        )
        annotator_head_metrics[label_name] = _classification_metrics(
            hard_labels[valid, doctor_index],
            doctor_predictions[valid, doctor_index],
            doctor_probabilities[valid, doctor_index],
            num_classes,
            class_names,
        )

    binary_probabilities = np.column_stack(
        (probabilities[:, 0], 1.0 - probabilities[:, 0])
    )
    result = {
        **{name: value / total_samples for name, value in loss_totals.items()},
        "mode_loss_weight": mode_loss_weight,
        "preds": predictions,
        "probs": probabilities,
        "annotator_probs": doctor_probabilities,
        "annotator_preds": doctor_predictions,
        "hard_labels": hard_labels,
        "consensus_labels": consensus_labels,
        "binary_probs": binary_probabilities,
        "binary_predictions": (predictions > 0).astype(int),
        "binary_consensus_labels": (consensus_labels > 0).astype(int),
        "patient_ids": np.asarray(patient_ids, dtype=object),
        "shared_pred_histogram": np.bincount(
            predictions, minlength=num_classes
        ),
        "annotator_pred_histogram": np.stack(
            [
                np.bincount(doctor_predictions[:, index], minlength=num_classes)
                for index in range(doctor_predictions.shape[1])
            ]
        ),
        "global_thresholds": global_thresholds.numpy(),
        "doctor_shifts": doctor_shifts.numpy(),
        "doctor_thresholds": doctor_thresholds.numpy(),
        "consensus_metrics": consensus_metrics,
    }
    for label_name, metrics in annotator_metrics.items():
        result[f"{label_name}_metrics"] = metrics
    for label_name, metrics in annotator_head_metrics.items():
        result[f"{label_name}_head_metrics"] = metrics
    return result
