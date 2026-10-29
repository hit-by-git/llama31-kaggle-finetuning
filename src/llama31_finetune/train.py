"""One DDP training experiment. Launched by run.py using torchrun."""
import argparse
import json
import math
import os
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, set_seed
from .config import load_config, experiment_dir
from .data import load_splits
from .modeling import load_base, load_tokenizer
from .prompting import CompletionCollator, encode_row
from .sampling import combine_tasks
from .provenance import split_fingerprints, prompt_contract


class TaskValidationTrainer(Trainer):
    """Average per-task answer loss while retaining each validation distribution."""
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        selected = self.eval_dataset if eval_dataset is None else eval_dataset
        if not isinstance(selected, dict):
            return super().evaluate(selected, ignore_keys, metric_key_prefix)
        metrics = {}
        for task, dataset in selected.items():
            metrics.update(super().evaluate(dataset, ignore_keys, f"{metric_key_prefix}_{task}"))
        metrics[f"{metric_key_prefix}_mean_task_loss"] = sum(
            metrics[f"{metric_key_prefix}_{task}_loss"] for task in selected) / len(selected)
        self.log({f"{metric_key_prefix}_mean_task_loss": metrics[f"{metric_key_prefix}_mean_task_loss"]})
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["tasks"] = args.tasks
    set_seed(cfg["seed"])
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    out = experiment_dir(cfg, args.run_name)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "adapter/adapter_config.json").exists() and not cfg["resume_from_checkpoint"]:
        raise FileExistsError(f"Completed adapter already exists: {out}. Use a new output_dir or explicit resume.")
    if cfg["resume_from_checkpoint"]:
        checkpoint = Path(cfg["resume_from_checkpoint"])
        previous = json.loads((checkpoint.parent.parent / "run_config.json").read_text())
        for key in ["model_id", "revision", "mode", "tasks", "seed", "max_length", "lora_r", "grad_accum", "num_gpus", "smoke_test", "data_root", "learning_rate", "epochs", "smoke_steps", "eval_steps"]:
            if previous[key] != cfg[key]:
                raise ValueError(f"Resume configuration differs: {key}")
    split_sets = {t: load_splits(cfg["data_root"], t, cfg["seed"]) for t in args.tasks}
    fingerprints = split_fingerprints(split_sets)
    if cfg["resume_from_checkpoint"]:
        saved_fingerprints = json.loads((checkpoint.parent.parent / "data_fingerprints.json").read_text())
        if saved_fingerprints != fingerprints:
            raise ValueError("Cannot resume: dataset fingerprints changed")
        if json.loads((checkpoint.parent.parent / "prompt_specs.json").read_text()) != prompt_contract(args.tasks):
            raise ValueError("Cannot resume: prompt definitions changed")
    tok = load_tokenizer(cfg)
    encoded, truncation, sizes = {}, {}, {}
    for split in ["train"]:
        mixed = combine_tasks({t: v[split] for t, v in split_sets.items()}, cfg["seed"],
                              balanced=(cfg["mode"] == "multitask"), smoke=cfg["smoke_test"])
        rows, truncated = [], 0
        for row in mixed.to_dict("records"):
            item, clipped = encode_row(row, tok, cfg["max_length"])
            rows.append(item); truncated += int(clipped)
        encoded[split] = Dataset.from_list(rows)
        truncation[split] = {"truncated": truncated, "total": len(rows)}
        sizes[split] = mixed.task.value_counts().to_dict()
    encoded["val"] = {}
    truncation["val"], sizes["val"] = {}, {}
    for task, parts in split_sets.items():
        frame = parts["val"]
        if cfg["smoke_test"]:
            frame = frame.groupby("label", group_keys=False).sample(
                n=min(16, int(frame.label.value_counts().min())), random_state=cfg["seed"])
        items, clipped_count = [], 0
        for row in frame.to_dict("records"):
            item, clipped = encode_row(row, tok, cfg["max_length"])
            items.append(item); clipped_count += int(clipped)
        encoded["val"][task] = Dataset.from_list(items)
        truncation["val"][task] = {"truncated": clipped_count, "total": len(items)}
        sizes["val"][task] = len(items)
    model = load_base(cfg, rank)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=2*cfg["lora_r"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if rank == 0:
        model.print_trainable_parameters()
        print("Answer-only supervision:", repr(tok.decode([v for v in rows[0]["labels"] if v != -100])))
        (out / "data_fingerprints.json").write_text(json.dumps(fingerprints, indent=2))
        (out / "run_config.json").write_text(json.dumps(cfg, indent=2))
        (out / "prompt_specs.json").write_text(json.dumps(prompt_contract(args.tasks), indent=2))
    interval = cfg["smoke_steps"] if cfg["smoke_test"] else cfg["eval_steps"]
    # Always evaluate/save even for a short real run.
    per_rank_examples = math.ceil(len(encoded["train"]) / cfg["num_gpus"])
    conservative_updates = max(1, per_rank_examples // cfg["grad_accum"])
    if not cfg["smoke_test"]:
        interval = min(interval, max(1, math.floor(conservative_updates * min(1, cfg["epochs"]))))
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=cfg["epochs"],
        max_steps=cfg["smoke_steps"] if cfg["smoke_test"] else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
        fp16=True, bf16=False, optim="paged_adamw_8bit", max_grad_norm=0.3,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=interval, save_strategy="steps", save_steps=interval,
        save_total_limit=cfg["save_total_limit"], load_best_model_at_end=True,
        metric_for_best_model="eval_mean_task_loss", greater_is_better=False, prediction_loss_only=True,
        logging_steps=1 if cfg["smoke_test"] else 10, ddp_find_unused_parameters=False,
        report_to="none", seed=cfg["seed"], data_seed=cfg["seed"], dataloader_num_workers=0,
        label_names=["labels"])
    trainer = TaskValidationTrainer(model=model, args=training_args, train_dataset=encoded["train"],
                      eval_dataset=encoded["val"], data_collator=CompletionCollator(tok), processing_class=tok)
    trainer.train(resume_from_checkpoint=cfg["resume_from_checkpoint"])
    trainer.save_model(str(out / "adapter"))
    if trainer.is_world_process_zero():
        tok.save_pretrained(out / "adapter")
        manifest = {"smoke_test": cfg["smoke_test"], "tasks": args.tasks,
                    "best_checkpoint": trainer.state.best_model_checkpoint,
                    "best_validation_loss": trainer.state.best_metric, "steps": trainer.state.global_step,
                    "model_commit": getattr(model.config, "_commit_hash", None),
                    "sample_counts": sizes, "truncation": truncation,
                    "data_fingerprints": split_fingerprints(split_sets),
                    "cleaning_audit": {t: v["train"].attrs.get("cleaning_audit", {}) for t, v in split_sets.items()}}
        (out / "training_summary.json").write_text(json.dumps(manifest, indent=2))
        (out / "training_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))


if __name__ == "__main__":
    main()
