"""Configuration without importing CUDA or model libraries."""
import json
from pathlib import Path

TASKS = ("misinfo", "hasoc", "constraint")
DEFAULTS = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct", "revision": "main",
    "mode": "separate", "tasks": list(TASKS), "seed": 42, "max_length": 512,
    "epochs": 2, "learning_rate": 1e-4, "lora_r": 16, "grad_accum": 8,
    "num_gpus": 2, "smoke_test": True, "smoke_steps": 10, "eval_steps": 50,
    "save_total_limit": 2, "resume_from_checkpoint": None,
}


def load_config(path):
    supplied = json.loads(Path(path).read_text())
    unknown = set(supplied) - set(DEFAULTS) - {"data_root", "output_dir"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    cfg = {**DEFAULTS, **supplied}
    if cfg["mode"] not in {"separate", "multitask"}:
        raise ValueError("mode must be separate or multitask")
    if not cfg["tasks"] or len(set(cfg["tasks"])) != len(cfg["tasks"]) or not set(cfg["tasks"]) <= set(TASKS):
        raise ValueError(f"tasks must be a nonempty, unique subset of {TASKS}")
    for name in ["data_root", "output_dir"]:
        if not cfg.get(name):
            raise ValueError(f"Missing {name}")
        cfg[name] = str(Path(cfg[name]).expanduser().resolve())
    for name in ["seed", "max_length", "lora_r", "grad_accum", "num_gpus", "smoke_steps", "eval_steps", "save_total_limit"]:
        if not isinstance(cfg[name], int) or cfg[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid {name}")
    if not isinstance(cfg["smoke_test"], bool):
        raise ValueError("smoke_test must be a boolean")
    if cfg["max_length"] < 256 or cfg["max_length"] % 8:
        raise ValueError("max_length must be >=256 and divisible by 8")
    if cfg["epochs"] <= 0 or cfg["learning_rate"] <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if cfg["resume_from_checkpoint"] and (cfg["mode"] != "multitask" and len(cfg["tasks"]) > 1):
        raise ValueError("Resume one separate task at a time, or a single multitask run")
    return cfg


def experiment_dir(cfg, run_name):
    return Path(cfg["output_dir"]) / ("smoke" if cfg["smoke_test"] else "full") / run_name


def run_groups(cfg):
    return [("multitask", cfg["tasks"])] if cfg["mode"] == "multitask" else [(t, [t]) for t in cfg["tasks"]]
