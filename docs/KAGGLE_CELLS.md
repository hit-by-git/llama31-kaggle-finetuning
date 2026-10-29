# Cells to paste into Kaggle

These cells clone the public repository directly. No GitHub token or source ZIP is needed. Attach only your data as a private Kaggle Dataset. The same cells are in `notebooks/kaggle_llama31.ipynb`.

## 1. Clone the public repository

In Kaggle enable Internet and select **T4 × 2**. Attach your original `data` folder as a private Kaggle Dataset, retaining `splits/`, `HASOC_eng/` and `constraint_aaai2021/`. The code repository is public, so cloning requires no GitHub token or source ZIP. Rerunning this cell pulls updates using a fast-forward merge; local code modifications cause Git to stop rather than overwrite them.

```python
from pathlib import Path
import zipfile, sys, subprocess, os, json

REPO_URL = "https://github.com/hit-by-git/llama31-kaggle-finetuning.git"
REPO = Path("/kaggle/working/llama31-kaggle-finetuning")
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
elif (REPO / ".git").is_dir():
    origin = subprocess.check_output(["git", "-C", str(REPO), "remote", "get-url", "origin"], text=True).strip()
    if origin != REPO_URL:
        raise ValueError(f"Existing folder belongs to a different repository: {origin}")
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
else:
    raise ValueError(f"Folder exists without Git metadata: {REPO}; select a new REPO path")
print("Project:", REPO)
print("Commit:", subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
```

## 2. Install dependencies

Run before importing model libraries. Keep Kaggle's PyTorch/CUDA build. Restart the session if these libraries were already imported; then rerun cells 1 and 3 onward. The local editable install has no additional dependency downloads.

```python
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-kaggle.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO)], check=True)
```

## 3. Load the Hugging Face secret and configure the experiment

Accept access to Llama 3.1 8B Instruct on your Hugging Face account. In Kaggle add a Secret named **HF_TOKEN**, enable it for this notebook, and load it below. Use a replacement for any token pasted into chat. No token belongs in source files.

Defaults train **three separate adapters**. Change `MODE` to `multitask` for one adapter using all datasets. Start with `TASKS=["misinfo"]` if session time is limited; HASOC is `hasoc`. Select the same tasks for smoke/full runs. The first run is a 10-step execution check, not reportable fine-tuning.

```python
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

MODE = "separate"  # "separate" or "multitask"
TASKS = ["misinfo", "hasoc", "constraint"]
SMOKE_TEST = True  # After the smoke run, set False and rerun cells 3, 4, 5 and 6.
DATA_ROOT = None   # e.g. Path("/kaggle/input/my-dataset/data")
OUTPUT_DIR = Path("/kaggle/working/llama31_outputs")  # Change for another experiment.

markers = {"misinfo": "splits/train.jsonl", "hasoc": "HASOC_eng/english_dataset.tsv",
           "constraint": "constraint_aaai2021/train.csv"}
if not TASKS or not set(TASKS) <= set(markers):
    raise ValueError("Select misinfo, hasoc, constraint, or a combination")
if DATA_ROOT is None:
    first = markers[TASKS[0]]
    candidates = sorted({p.parent.parent for p in Path("/kaggle/input").rglob(Path(first).name)
                         if str(p).endswith(first)})
    candidates = [p for p in candidates if all((p / markers[t]).exists() for t in TASKS)]
    if len(candidates) != 1:
        raise ValueError(f"Set DATA_ROOT explicitly; matching roots: {candidates}")
    DATA_ROOT = candidates[0]
config = json.loads((REPO / "configs/example.json").read_text())
config.update(mode=MODE, tasks=TASKS, smoke_test=SMOKE_TEST,
              data_root=str(DATA_ROOT), output_dir=str(OUTPUT_DIR))
CONFIG = Path("/kaggle/working/llama31_run.json")
CONFIG.write_text(json.dumps(config, indent=2))
# Configuration contains no secrets.
print(json.dumps(config, indent=2))
```

## 4. Check data, environment and model access

The audit reports cleanup and split sizes. Preflight checks CUDA GPUs, package imports and gated tokenizer access; the next cell verifies the full model download and training. Misinfo has 2304/495/489 rows; HASOC 4941/873/1153; CONSTRAINT 6417/2140/2140 after documented cleanup.

```python
subprocess.run([sys.executable, "-m", "llama31_finetune.run", "--config", str(CONFIG), "--stage", "prepare"], check=True)
subprocess.run([sys.executable, "-m", "llama31_finetune.preflight", "--config", str(CONFIG)], check=True)
print((OUTPUT_DIR / "prepared/data_audit.json").read_text())
```

## 5. Train

Run the smoke test first. Check its answer-only supervision output and that training/evaluation complete. Then set `SMOKE_TEST=False` in cell 3 and rerun cells 3–6. Full runs use different paths, so they cannot load a smoke adapter accidentally. Three separate adapters run sequentially and may require separate Kaggle sessions; choose one task at a time if needed.

For OOM reduce `max_length` in `config` to 384 or 256 before writing `CONFIG`; inspect truncation. Then consider `lora_r=8`. Existing outputs are protected; use a new `OUTPUT_DIR` for a fresh experiment. To resume, select one separate task, set `resume_from_checkpoint` to its checkpoint, and retain matching run metadata and settings.

```python
subprocess.run([sys.executable, "-m", "llama31_finetune.run", "--config", str(CONFIG), "--stage", "train"], check=True)
```

## 6. Evaluate validation and inspect results

Smoke validation uses a small subset to check execution; full validation uses every validation example. Each task compares majority, TF-IDF, original Llama and the fine-tuned adapter. Choose settings with validation only. The training checkpoint is selected by mean per-task validation answer loss, a proxy for macro-F1. Examine generated validation scores before reporting final tests.

```python
cmd = [sys.executable, "-m", "llama31_finetune.run", "--config", str(CONFIG),
       "--stage", "evaluate", "--split", "val"]
if SMOKE_TEST:
    cmd += ["--max-rows", "32"]
subprocess.run(cmd, check=True)

import pandas as pd
from IPython.display import display, Image
RUN_ROOT = OUTPUT_DIR / ("smoke" if SMOKE_TEST else "full")
records = []
for path in RUN_ROOT.glob("*/evaluation/val/*/metrics_*.json"):
    result = json.loads(path.read_text())
    records.append({"adapter": path.parents[3].name, "task": path.parent.name,
                    "model": path.stem.removeprefix("metrics_"),
                    **{k: result[k] for k in ["n", "accuracy", "macro_f1", "weighted_f1", "invalid_rate"]}})
display(pd.DataFrame(records))
for path in RUN_ROOT.glob("*/evaluation/val/*/confusion_finetuned.png"):
    display(Image(filename=str(path)))
```

## 7. Final test — run after fixing settings

Set `RUN_FINAL_TEST=True` only after finalizing settings on validation. This cell refuses to label a smoke adapter as a final run. Use 500 bootstrap samples for 95% intervals; article–post intervals resample event groups. Report each task separately, plus language/variant/tactic slices, invalid outputs and truncation. Read `docs/IMPLEMENTATION_PLAN.md` and README for the known annotation concerns. Repeated test-driven tuning makes the test set a development set.

```python
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    if SMOKE_TEST:
        raise ValueError("Run full training before the final test")
    subprocess.run([sys.executable, "-m", "llama31_finetune.run", "--config", str(CONFIG),
                    "--stage", "evaluate", "--split", "test", "--bootstrap", "500"], check=True)
    records = []
    for path in (OUTPUT_DIR / "full").glob("*/evaluation/test/*/metrics_*.json"):
        result = json.loads(path.read_text())
        records.append({"adapter": path.parents[3].name, "task": path.parent.name,
                        "model": path.stem.removeprefix("metrics_"),
                        **{k: result[k] for k in ["n", "accuracy", "macro_f1", "invalid_rate"]}})
    display(pd.DataFrame(records))
else:
    print("Final test skipped; use validation until the experiment is fixed.")
```

## 8. Save adapters and reports

The archive excludes optimizer checkpoints to keep it smaller. Download checkpoints separately if you want exact training resume. Adapters still need the gated base model. Save a Kaggle notebook version with outputs or download the ZIP before the session ends. Prediction files include dataset text; keep them private if required.

```python
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(OUTPUT_DIR / "environment.txt").write_text(freeze)
archive = Path("/kaggle/working/llama31_adapters_and_reports.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(CONFIG, "launch_config.json")
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and "checkpoints" not in p.parts:
            z.write(p, p.relative_to(OUTPUT_DIR))
print("Saved:", archive, "MB:", round(archive.stat().st_size / 1e6, 1))
```
