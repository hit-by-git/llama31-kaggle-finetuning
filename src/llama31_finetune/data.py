"""Dataset loading and leakage checks; imports no model or GPU packages."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TASK_SPECS = {
    "misinfo": {"fields": ["article", "post"], "labels": ["not_misinfo", "misinfo"]},
    "hasoc": {"fields": ["text"], "labels": ["NOT", "HOF"]},
    "constraint": {"fields": ["tweet"], "labels": ["real", "fake"]},
}


def normalize_input(value: str) -> str:
    """Normalize only for duplicate detection; model inputs are unchanged."""
    return " ".join(str(value).casefold().split())


def _input_keys(frame: pd.DataFrame, fields: list[str]) -> pd.Series:
    return frame[fields].apply(
        lambda row: tuple(normalize_input(value) for value in row), axis=1
    )


def validate_splits(parts: dict[str, pd.DataFrame], task: str) -> None:
    """Reject invalid labels, empty fields and cross-split input/event leakage."""
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task {task!r}; choose {list(TASK_SPECS)}")
    if set(parts) != {"train", "val", "test"}:
        raise ValueError("Expected exactly train, val and test splits")
    spec = TASK_SPECS[task]
    keys = {}
    for split, frame in parts.items():
        required = spec["fields"] + ["label"]
        if task == "misinfo":
            required += ["event_id"]
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {task}/{split}: {sorted(missing)}")
        if frame.empty or frame[required].isna().any().any():
            raise ValueError(f"Empty split or missing input/label in {task}/{split}")
        actual = set(frame["label"])
        expected = set(spec["labels"])
        if actual != expected:
            raise ValueError(
                f"Expected both classes {sorted(expected)} in {task}/{split}; got {sorted(actual)}"
            )
        for field in spec["fields"]:
            if frame[field].astype(str).str.strip().eq("").any():
                raise ValueError(f"Empty {field} in {task}/{split}")
        keys[split] = set(_input_keys(frame, spec["fields"]))
    for a, b in combinations(parts, 2):
        if keys[a].intersection(keys[b]):
            raise ValueError(f"Duplicate inputs across {task} {a}/{b}; resolve leakage first")
        if task == "misinfo":
            events_a = set(parts[a]["event_id"].astype(str))
            events_b = set(parts[b]["event_id"].astype(str))
            if events_a.intersection(events_b):
                raise ValueError(f"Event leakage across {a}/{b}")


def load_splits(data_root: str | Path, task: str, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Load one task, preserve official tests and expose a cleaning audit in attrs.

    Each result has a ``task`` column and ``attrs['cleaning_audit']`` containing
    row counts and exclusions. Original files are never modified. Duplicate
    cleanup uses inputs only, never test labels to revise training labels.
    """
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task {task!r}; choose {list(TASK_SPECS)}")
    root = Path(data_root)
    audit = {"task": task, "seed": seed, "exclusions": {}}
    if task == "misinfo":
        parts = {
            split: pd.read_json(root / "splits" / f"{split}.jsonl", lines=True)
            for split in ["train", "val", "test"]
        }
    elif task == "constraint":
        parts = {
            split: pd.read_csv(root / "constraint_aaai2021" / f"{filename}.csv")
            for split, filename in [("train", "train"), ("val", "validation"), ("test", "test")]
        }
        audit["original_rows"] = {name: len(frame) for name, frame in parts.items()}
        seen = set()
        for split in ["test", "val", "train"]:
            frame = parts[split]
            # Do not let missing fields disappear as an apparent duplicate.
            if "tweet" not in frame or frame["tweet"].isna().any():
                raise ValueError(f"Missing tweet in constraint/{split}")
            normalized = frame["tweet"].map(normalize_input)
            keep = ~normalized.isin(seen)
            audit["exclusions"][f"{split}_cross_split_duplicates"] = int((~keep).sum())
            parts[split] = frame.loc[keep].copy()
            seen.update(normalized.loc[keep])
    else:
        directory = root / "HASOC_eng"
        train = pd.read_csv(directory / "english_dataset.tsv", sep="\t").rename(
            columns={"task_1": "label"}
        )
        test = pd.read_csv(directory / "hasoc2019_en_test-2919.tsv", sep="\t").rename(
            columns={"task_1": "label"}
        )
        if train[["text", "label"]].isna().any().any():
            raise ValueError("Missing text or label in HASOC training data")
        audit["original_rows"] = {"train": len(train), "test": len(test)}
        keyed = train.assign(_input_key=train["text"].map(normalize_input))
        labels_per_key = keyed.groupby("_input_key")["label"].nunique()
        ambiguous = set(labels_per_key[labels_per_key > 1].index)
        conflict_mask = keyed["_input_key"].isin(ambiguous)
        audit["exclusions"]["conflicting_duplicate_groups"] = len(ambiguous)
        audit["exclusions"]["conflicting_duplicate_rows"] = int(conflict_mask.sum())
        clean = keyed.loc[~conflict_mask].copy()
        before = len(clean)
        clean = clean.drop_duplicates("_input_key")
        audit["exclusions"]["same_label_duplicate_rows"] = before - len(clean)
        train, val = train_test_split(
            clean, test_size=0.15, stratify=clean["label"], random_state=seed
        )
        parts = {
            "train": train.drop(columns="_input_key"),
            "val": val.drop(columns="_input_key"),
            "test": test,
        }
    if "original_rows" not in audit:
        audit["original_rows"] = {name: len(frame) for name, frame in parts.items()}
    validate_splits(parts, task)
    audit["final_rows"] = {name: len(frame) for name, frame in parts.items()}
    result = {}
    for split, frame in parts.items():
        frame = frame.reset_index(drop=True).copy()
        for field in TASK_SPECS[task]["fields"]:
            frame[field] = frame[field].astype(str)
        frame["task"] = task
        frame.attrs["cleaning_audit"] = audit
        result[split] = frame
    return result
