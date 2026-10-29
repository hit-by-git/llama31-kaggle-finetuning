# Implementation plan and status

The initial experiment trains three separate LoRA adapters on the same Llama 3.1 8B Instruct base. An optional multitask run learns all three tasks in one adapter using task-specific prompts and deterministic task-balanced training. The two modes require separate experiments and output paths.

1. **Data preparation — implemented and locally verified.** Preserve official test files, retain event-separated misinformation splits, remove cross-split repeated CONSTRAINT training inputs, exclude ambiguous HASOC training duplicates, and report every exclusion. Keep source data outside Git. Save hashes of every cleaned split.
2. **Training — implemented; GPU verification pending.** Frozen 4-bit NF4 base, FP16 computation, rank-16 LoRA, one model replica per T4, microbatch 1, gradient accumulation 8, 512 tokens, gradient checkpointing, and two epochs. Task-balanced sampling duplicates smaller task training sets without discarding their examples. Validation distributions remain natural within each task; checkpoint selection uses the equal-weight mean of per-task answer losses as a proxy for classification performance.
3. **Validation — implemented; GPU execution pending.** Run the 10-step smoke test before full training. Generate validation labels and compare separate baselines. Change settings using validation only. A second or subsequent run must use a fresh output directory unless explicitly resuming.
4. **Final evaluation — implemented; GPU execution pending.** Evaluate the frozen original Instruct model, fine-tuned adapter, majority classifier and TF-IDF classifier on exactly the same test rows. Report each task separately: macro-F1, accuracy, weighted F1, per-class precision/recall/support, confusion matrix, invalid-output rate, inference throughput, peak allocated GPU memory and truncation. Include language/variant/tactic slices and optional 95% bootstrap intervals. Bootstrap events for the article/post task.
5. **Reproducibility and packaging — implemented.** Saved config, prompt/label contract, model commit, cleaned split hashes, audit and training history accompany adapters and reports. Smoke/full outputs are isolated. Credentials, datasets, model weights and generated predictions are excluded from Git.
6. **Repository delivery — local implementation complete; remote creation/push performed after checks.** Use a new nested repository rather than modifying the existing parent repository. Default destination is private `hit-by-git/llama31-kaggle-finetuning`.

## Experiment order

- Smoke test one task first (`tasks=["misinfo"]`); then smoke all selected tasks if needed.
- Full separate adapters, starting with the task of primary interest. Run tasks in separate Kaggle sessions if GPU quota/time is insufficient.
- Compare generated validation scores before choosing settings.
- Optionally run multitask with `mode="multitask"` in a fresh output directory.
- Final test once the procedure is fixed; report three task-specific scorecards.

## Limits and data audit

The supplied misinformation data has suspected labeling inconsistencies: some nominally neutral posts change the reference article's weekday. Human review is required before treating benchmark scores as reliable. This project does not silently relabel them. HASOC `HOF` combines hate and offensive language; CONSTRAINT tweet-only predictions are not evidence-based fact checking. Agreement with synthetic reference pairs does not establish real-world truth verification.

CPU tests validate behavior and data preparation; they do not establish CUDA/bitsandbytes compatibility or prove that the model fits on a particular Kaggle image. Preflight checks imports/access and the smoke run verifies training on the actual GPUs. Runtime, quota, package compatibility and disk usage depend on the Kaggle environment.
