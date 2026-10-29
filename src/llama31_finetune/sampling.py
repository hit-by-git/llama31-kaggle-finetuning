"""Deterministic per-task sampling; no model dependencies."""
import pandas as pd


def combine_tasks(frames, seed=42, balanced=False, smoke=False):
    if not frames or any(frame.empty for frame in frames.values()):
        raise ValueError("Every selected task needs a nonempty training frame")
    selected = []
    for frame in frames.values():
        if smoke:
            # Preserve both classes even in the execution check.
            frame = frame.groupby("label", group_keys=False).sample(
                n=min(16, int(frame.label.value_counts().min())), random_state=seed)
        selected.append(frame.copy())
    target = max(map(len, selected))
    if balanced:
        # Keep every original example, and duplicate only the smaller task's rows.
        selected = [pd.concat([frame, frame.sample(n=target-len(frame), replace=True, random_state=seed)],
                              ignore_index=True) if len(frame) < target else frame for frame in selected]
    return pd.concat(selected, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
