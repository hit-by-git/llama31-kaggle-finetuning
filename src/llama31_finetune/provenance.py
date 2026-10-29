"""Portable fingerprints for input content and the run's inference contract."""
import hashlib
import json
from .data import TASK_SPECS
from .prompting import INSTRUCTIONS


def split_fingerprints(split_sets):
    result = {}
    for task, parts in split_sets.items():
        fields = TASK_SPECS[task]["fields"] + ["label"]
        if task == "misinfo":
            fields += ["event_id"]
        result[task] = {}
        for split, frame in parts.items():
            records = frame[fields].astype(str).to_dict("records")
            payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            result[task][split] = hashlib.sha256(payload.encode()).hexdigest()
    return result


def prompt_contract(tasks):
    return {task: {**TASK_SPECS[task], "instruction": INSTRUCTIONS[task],
                   "codes": {"A": TASK_SPECS[task]["labels"][0], "B": TASK_SPECS[task]["labels"][1]}}
            for task in tasks}
