"""CPU orchestration: data checks, separate torchrun workers, and evaluation."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .config import load_config, run_groups, experiment_dir
from .data import load_splits
from .provenance import split_fingerprints


def prepare(cfg):
    prepared = Path(cfg["output_dir"]) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    audits = {}
    parts = {}
    for task in cfg["tasks"]:
        splits = load_splits(cfg["data_root"], task, cfg["seed"])
        parts[task] = splits
        audits[task] = splits["train"].attrs["cleaning_audit"]
        print(task, {name: len(frame) for name, frame in splits.items()})
    (prepared / "data_audit.json").write_text(json.dumps(audits, indent=2))
    (prepared / "data_fingerprints.json").write_text(json.dumps(split_fingerprints(parts), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "prepare":
        prepare(cfg)
        return
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "WANDB_DISABLED": "true",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for run_name, tasks in run_groups(cfg):
        common = ["--config", str(Path(args.config).resolve()), "--run-name", run_name, "--tasks", *tasks]
        if args.stage == "train":
            out = experiment_dir(cfg, run_name)
            if out.exists() and any(out.iterdir()) and not cfg["resume_from_checkpoint"]:
                raise FileExistsError(f"Run output already exists: {out}; use a new output_dir or resume explicitly")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={cfg['num_gpus']}", "--module", "llama31_finetune.train", *common]
        else:
            command = [sys.executable, "-m", "llama31_finetune.evaluate", *common,
                       "--split", args.split, "--bootstrap", str(args.bootstrap)]
            if args.max_rows is not None:
                command += ["--max-rows", str(args.max_rows)]
        print("Starting", args.stage, run_name, flush=True)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
