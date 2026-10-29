# Llama 3.1 8B QLoRA on Kaggle T4 × 2

Fine-tune `meta-llama/Llama-3.1-8B-Instruct` on the three supplied classification datasets, then compare generated labels with held-out labels. Start with **three separate adapters**. Set `mode="multitask"` to train one adapter with all three task-specific prompts. Both modes use the same per-task evaluation protocol.

| Task | Input | Labels | Clean train / validation / test |
|---|---|---|---|
| `misinfo` | article + post | not_misinfo / misinfo | 2304 / 495 / 489 |
| `hasoc` | tweet | NOT / HOF | 4941 / 873 / 1153 |
| `constraint` | tweet | real / fake | 6417 / 2140 / 2140 |

The counts above describe the uploaded files after documented cleanup, with seed 42. Task labels retain their own meanings. HASOC HOF includes offense/profanity. Article/post classification tests consistency with the reference article. CONSTRAINT classifies its COVID-19 tweets without external evidence.

## Run on Kaggle

Use [the notebook](notebooks/kaggle_llama31.ipynb), or copy the numbered cells from [KAGGLE_CELLS.md](docs/KAGGLE_CELLS.md). Both follow the same procedure:

1. Select T4 × 2, enable Internet, and attach the uploaded `data` folder as a private Kaggle Dataset.
2. Run the first notebook cell to clone this public repository directly. Rerunning it pulls updates with `git pull --ff-only`. No GitHub token or source ZIP is needed.
3. Accept the [Meta model license/access terms](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Add a valid read token as Kaggle Secret **HF_TOKEN**, enabled for the notebook. Revoke any token previously pasted into chat; do not put the replacement in a cell.
4. Clone the source, install pinned packages, set the data root, run preflight and data checks, then the 10-step smoke run.
5. Use `smoke_test=false` for the full run. Evaluate validation first. Once settings are final, run the separate final-test cell.
6. Save Version with notebook outputs, or download the adapter/report archive before the session ends.

A T4 × 2 setup has two independent GPU memories. The workers each hold a complete 4-bit model replica; `device_map="auto"` is not used for DDP training. Defaults: FP16, 512 tokens, microbatch 1 per GPU, accumulation 8 (global batch 16), rank-16 LoRA, two epochs. This is adapter training, not full updates to all 8B weights.

Training all three separate adapters is sequential and can exceed a free session's available time. Choose one task per session as needed. A multitask epoch includes duplicated examples from smaller tasks; its total work differs from separate runs. No runtime estimate is promised.

## CLI

Install `requirements-kaggle.txt` and `pip install --no-deps -e .`, edit a copy of `configs/example.json`, then:

```bash
python -m llama31_finetune.run --config run.json --stage prepare
python -m llama31_finetune.preflight --config run.json
python -m llama31_finetune.run --config run.json --stage train
python -m llama31_finetune.run --config run.json --stage evaluate --split val
# After fixing the experiment using validation only:
python -m llama31_finetune.run --config run.json --stage evaluate --split test --bootstrap 500
```

The parent runner stays on CPU and launches fresh torchrun or evaluation workers. GPU memory is released between experiments. Change `num_gpus=1` if only one compatible GPU is available; global batch then halves. Lower `max_length` to 384/256 for OOM, inspect truncation, then consider `lora_r=8`. Accumulation does not reduce the memory of one example.

Outputs: `output_dir/{smoke|full}/{task|multitask}/adapter`, `checkpoints`, run metadata, and `evaluation/{val|test}/{task}/` reports/predictions/confusion plots. Existing run/evaluation outputs are protected from silent overwrite. For another experiment, change `output_dir`. For an intentional evaluation rerun, archive the old evaluation directory first.

Resume by selecting exactly one separate task (or the matching multitask run) and setting `resume_from_checkpoint` to an existing `checkpoints/checkpoint-N`. Keep the same task, data, mode, seed, GPU count, sequence length and training settings; config and fingerprints are checked. A copied checkpoint must retain its run metadata two directories above it. An adapter alone has no optimizer state and cannot resume the full training state.

## Evaluation and reproducibility

Report **macro-F1** for each task, every class's precision/recall, accuracy, confusion matrix, and invalid-output rate. The evaluation compares original Llama Instruct, fine-tuned adapter, majority and TF-IDF logistic regression on identical rows with deterministic label generation. Invalid text/refusals count as errors. No perplexity/BLEU/ROUGE claim is used as a classification score.

Checkpoint selection averages validation answer loss across tasks, retaining each task's natural validation distribution. At completion, rank 0 exports the already-saved best adapter directly; workers do not reload it through Trainer's conditional end-of-training distributed barrier. This is a proxy: use generated validation macro-F1 to select final settings and compare modes. Final test results must not guide further tuning. Label-preserving task balancing applies only to training.

Optional bootstrap intervals resample entire `event_id` groups for misinformation and rows for the other tasks. They quantify sampling uncertainty for a fixed trained model, not training-seed variance. Repeat seeds if resources allow. Subgroups missing a class receive a caveat; use counts/accuracy instead of comparing their fixed-label F1 to overall F1.

Saved manifests record input hashes, actual model commit, config, prompts, label mappings, cleanup and truncation. Evaluation uses saved inference settings and rejects changed input content/prompts. Load the original gated model plus the adapter; an adapter is not a standalone model. Keep reports private if their text is sensitive.

The supplied misinformation labels need an audit: some neutral posts appear to change weekdays in the source. Original files are preserved. Cleanup removes 27 ambiguous HASOC training rows plus 11 same-label duplicates, and 3 CONSTRAINT training repeats that occur in other splits. See `prepared/data_audit.json` for exact exclusions.

## Local checks

```bash
python -m pip install -r requirements-test.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

CPU tests cover data leakage, cleanup, metrics, grouped bootstrap, strict completion masking, truncation, task-specific prompts, sampling and output isolation. Full GPU training has not been executed in this local workspace. The notebook includes actual-environment preflight and smoke checks.

See [implementation plan](docs/IMPLEMENTATION_PLAN.md) for the experiment sequence and limitations. Pinned packages target [Transformers 4.51.3 Trainer](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer) and [PEFT quantization](https://huggingface.co/docs/peft/v0.15.0/developer_guides/quantization). Keep Kaggle's PyTorch/CUDA stack; restart after changing already-imported libraries and resolve any incompatible optional torchvision/torchaudio installation before training.

## If a worker stalls after validation

A 5–10 minute pause after a synchronization warning needs investigation. A barrier warning alone does not identify the root cause. The training entry point now logs checkpoint start/end, adapter export and worker exit, disables the conditional best-model reload at the end, and uses a five-minute distributed collective timeout. The same validation-selected checkpoint is exported without GPU reloading. This workaround still needs a new Kaggle GPU smoke run.

Stop the running cell before pulling updated code. Keep the current session and existing checkpoint directories. Rerun cell 1 to pull, then use `TASKS=["misinfo"]`, `SMOKE_TEST=True`, and a fresh `OUTPUT_DIR` (for example `/kaggle/working/llama31_outputs_retry1`) in cell 3. Run cells 3–5. No dataset reupload or dependency reinstall is needed for this source-only update. If old worker processes remain, inspect GPU/process activity before launching another run. Do not delete old outputs to bypass the overwrite guard.
