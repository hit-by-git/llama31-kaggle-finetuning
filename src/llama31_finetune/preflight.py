"""Fail early on missing GPU, dependencies or gated-model access."""
import argparse
import os
import importlib.metadata
from .config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    import torch
    import bitsandbytes
    from transformers import Trainer
    from peft import LoraConfig
    from huggingface_hub import hf_hub_download
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Add HF_TOKEN to Kaggle Secrets and load it into the environment")
    if torch.cuda.device_count() < cfg["num_gpus"]:
        raise RuntimeError(f"Need {cfg['num_gpus']} CUDA GPUs; found {torch.cuda.device_count()}")
    for i in range(cfg["num_gpus"]):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}; {props.total_memory/1024**3:.1f} GiB")
    hf_hub_download(cfg["model_id"], "tokenizer_config.json", revision=cfg["revision"], token=os.environ["HF_TOKEN"])
    for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"]:
        print(name, importlib.metadata.version(name))
    print("Preflight passed. The full model download and GPU smoke test are still required.")


if __name__ == "__main__":
    main()
