"""Classification reports with invalid generations counted as errors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


INVALID = "INVALID"


def _basic_scores(y: Sequence[str], predictions: Sequence[str], labels: list[str]) -> dict:
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predictions, labels=labels, average="weighted", zero_division=0)),
        "invalid_rate": float(np.mean(np.asarray(predictions) == INVALID)),
        "class_support": {label: int(np.sum(np.asarray(y) == label)) for label in labels},
    }


def score_predictions(
    frame: pd.DataFrame,
    predictions: Sequence[str],
    labels: Sequence[str],
    bootstrap: int = 0,
    seed: int = 42,
) -> dict:
    """Score generated labels, keeping metrics separate for each task.

    Predictions outside ``labels`` become INVALID and are never dropped. Macro
    F1 averages only the real classes; invalid outputs produce false negatives.
    If event IDs exist, confidence intervals resample complete event groups;
    otherwise they resample rows. Intervals describe sampling uncertainty for a
    fixed trained model, not training-seed variability.
    """
    labels = list(labels)
    if not labels or len(labels) != len(set(labels)) or INVALID in labels:
        raise ValueError("Provide distinct real labels, excluding INVALID")
    if frame.empty:
        raise ValueError("Cannot score an empty frame")
    if "label" not in frame or frame["label"].isna().any():
        raise ValueError("Every example must have a true label")
    if len(predictions) != len(frame):
        raise ValueError("Prediction count must equal the number of examples")
    if not set(frame["label"]).issubset(labels):
        raise ValueError("Unknown true labels")
    if not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    y = frame["label"].astype(str).to_numpy()
    predicted = np.asarray([p if isinstance(p, str) and p in labels else INVALID for p in predictions])
    report = _basic_scores(y, predicted, labels)
    report["labels"] = labels
    report["per_class"] = classification_report(
        y, predicted, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y, predicted, labels=labels + [INVALID])
    report["confusion_matrix"] = {
        "true_labels": labels,
        "predicted_labels": labels + [INVALID],
        "counts": matrix[:len(labels), :].tolist(),
    }
    report["subgroups"] = {}
    for field in ["language", "variant", "tactic"]:
        if field not in frame:
            continue
        slices = {}
        # groupby.indices gives positions even when the DataFrame has a custom index.
        for value, index in frame.groupby(field, dropna=False, sort=False).indices.items():
            idx = np.asarray(index, dtype=int)
            item = _basic_scores(y[idx], predicted[idx], labels)
            item["present_classes"] = sorted(set(y[idx]))
            if len(item["present_classes"]) < len(labels):
                item["note"] = (
                    "This slice lacks a true class. Macro-F1 uses the full fixed label set; "
                    "compare counts and accuracy rather than comparing this F1 with overall F1."
                )
            slices["<missing>" if pd.isna(value) else str(value)] = item
        report["subgroups"][field] = slices
    if bootstrap:
        if "event_id" in frame:
            if frame["event_id"].isna().any():
                raise ValueError("Missing event_id prevents grouped bootstrap")
            groups = [np.asarray(idx, dtype=int) for idx in frame.groupby("event_id", sort=False).indices.values()]
            unit = "event_id"
        else:
            groups = [np.asarray([index]) for index in range(len(frame))]
            unit = "row"
        rng = np.random.default_rng(seed)
        values = {"accuracy": [], "macro_f1": []}
        for _ in range(bootstrap):
            selected = rng.integers(0, len(groups), size=len(groups))
            index = np.concatenate([groups[int(i)] for i in selected])
            values["accuracy"].append(float(accuracy_score(y[index], predicted[index])))
            values["macro_f1"].append(float(f1_score(
                y[index], predicted[index], labels=labels, average="macro", zero_division=0
            )))
        report["confidence_intervals"] = {
            "confidence_level": 0.95,
            "method": "percentile_bootstrap",
            "resampling_unit": unit,
            "n_units": len(groups),
            "n_resamples": bootstrap,
            "seed": seed,
            "metrics": {
                name: {"lower": float(np.quantile(scores, 0.025)), "upper": float(np.quantile(scores, 0.975))}
                for name, scores in values.items()
            },
            "note": "Intervals measure test-sampling uncertainty for this fixed model, not training-seed variation.",
        }
    return report
