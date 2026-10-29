"""Evaluate fixed validation/test inputs in a fresh, single-GPU process."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from .config import load_config, experiment_dir
from .data import load_splits, TASK_SPECS
from .metrics import score_predictions
from .modeling import load_base, load_tokenizer
from .prompting import make_prompt, label_maps
from .provenance import split_fingerprints, prompt_contract


def save_report(directory, name, frame, predictions, bootstrap, seed, extras=None, metadata=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    task = frame.task.iloc[0]
    scores = score_predictions(frame, predictions, TASK_SPECS[task]["labels"], bootstrap, seed)
    scores["run"] = metadata or {}
    (directory / f"metrics_{name}.json").write_text(json.dumps(scores, indent=2))
    output = frame.copy()
    output["prediction"] = predictions
    for key, values in (extras or {}).items():
        output[key] = values
    output.to_csv(directory / f"predictions_{name}.csv", index=False)
    cm = scores["confusion_matrix"]
    values = np.array(cm["counts"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(values, cmap="Blues")
    ax.set(xticks=range(values.shape[1]), xticklabels=cm["predicted_labels"],
           yticks=range(values.shape[0]), yticklabels=cm["true_labels"],
           xlabel="Predicted", ylabel="True", title=f"{task}: {name}")
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout(); fig.savefig(directory / f"confusion_{name}.png", dpi=150); plt.close(fig)
    print(task, name, {k: scores[k] for k in ["n", "accuracy", "macro_f1", "invalid_rate"]}, flush=True)
    return scores


@torch.inference_mode()
def predict(model, tokenizer, cfg, frame):
    preds, answers, truncation = [], [], []
    model.eval()
    model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i, row in enumerate(frame.to_dict("records")):
        _, ids, truncated = make_prompt(row, tokenizer, cfg["max_length"])
        inputs = torch.tensor([ids], device="cuda:0")
        generated = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=8,
            do_sample=False, eos_token_id=list(set([tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")])), pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(generated[0, inputs.shape[1]:], skip_special_tokens=True).strip()
        _, code_to_label = label_maps(row["task"])
        preds.append(code_to_label.get(answer, "INVALID"))
        answers.append(answer); truncation.append(truncated)
        if (i + 1) % 50 == 0:
            print(f"Evaluated {i+1}/{len(frame)}", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    extras = {"raw_answer": answers, "input_truncated": truncation}
    timing = {"seconds": seconds, "examples_per_second": len(frame) / seconds,
              "seconds_per_example": seconds / len(frame),
              "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
              "truncated_inputs": int(sum(truncation)),
              "timing_note": "Includes prompt tokenization and inference, excludes model loading; batch size 1."}
    return preds, extras, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()
    requested = load_config(args.config)
    out = experiment_dir(requested, args.run_name)
    cfg = json.loads((out / "run_config.json").read_text())
    # Use the saved inference settings, allowing only a relocated identical dataset.
    cfg["data_root"] = requested["data_root"]
    manifest = json.loads((out / "training_summary.json").read_text())
    if args.tasks != manifest["tasks"]:
        raise ValueError("Evaluation tasks differ from trained run")
    contract = json.loads((out / "prompt_specs.json").read_text())
    if contract != prompt_contract(args.tasks):
        raise ValueError("Prompt code changed since training; restore the original code")
    parts = {task: load_splits(cfg["data_root"], task, cfg["seed"]) for task in args.tasks}
    if split_fingerprints(parts) != manifest["data_fingerprints"]:
        raise ValueError("Dataset content differs from training manifest")
    if manifest.get("model_commit"):
        cfg["revision"] = manifest["model_commit"]
    eval_frames = {}
    for task in args.tasks:
        frame = parts[task][args.split]
        if args.max_rows is not None:
            if args.max_rows < 1:
                raise ValueError("--max-rows must be positive")
            frame = frame.sample(min(args.max_rows, len(frame)), random_state=cfg["seed"]).reset_index(drop=True)
        eval_frames[task] = frame
    metadata = {"split": args.split, "smoke_test": cfg["smoke_test"],
                "subset": args.max_rows is not None, "seed": cfg["seed"], "run_name": args.run_name}
    # Do not silently overwrite a previous final test assessment.
    report_root = out / "evaluation" / args.split
    if report_root.exists():
        raise FileExistsError(f"Evaluation exists: {report_root}. Archive it before an intentional rerun.")
    report_root.mkdir(parents=True)
    summary = {}
    for task, frame in eval_frames.items():
        destination = report_root / task
        train = parts[task]["train"]
        summary[task] = {}
        majority = train.label.value_counts().idxmax()
        summary[task]["majority"] = save_report(destination, "majority", frame, [majority]*len(frame),
                                                     args.bootstrap, cfg["seed"], metadata=metadata)
        fields = TASK_SPECS[task]["fields"]
        def texts(f):
            return f[fields].apply(lambda row: " [SEP] ".join(row), axis=1).tolist()
        baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
                                 LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg["seed"]))
        baseline.fit(texts(train), train.label)
        summary[task]["tfidf"] = save_report(destination, "tfidf", frame, baseline.predict(texts(frame)),
                                                  args.bootstrap, cfg["seed"], metadata=metadata)
    tokenizer = load_tokenizer(cfg)
    model = load_base(cfg, 0)
    for task, frame in eval_frames.items():
        preds, extras, timing = predict(model, tokenizer, cfg, frame)
        summary[task]["original"] = save_report(report_root / task, "original", frame, preds,
            args.bootstrap, cfg["seed"], extras, {**metadata, **timing})
    model = PeftModel.from_pretrained(model, str(out / "adapter"), is_trainable=False)
    for task, frame in eval_frames.items():
        preds, extras, timing = predict(model, tokenizer, cfg, frame)
        summary[task]["finetuned"] = save_report(report_root / task, "finetuned", frame, preds,
            args.bootstrap, cfg["seed"], extras, {**metadata, **timing})
    (report_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Reports:", report_root)


if __name__ == "__main__":
    main()
