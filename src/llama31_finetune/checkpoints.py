"""Export an already-saved LoRA checkpoint without loading it on the GPUs."""
import json
import shutil
import tempfile
from pathlib import Path


def export_adapter(checkpoint, destination):
    """Atomically copy adapter files; retain optimizer state in the checkpoint."""
    source, destination = Path(checkpoint), Path(destination)
    config = source / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Adapter configuration missing: {config}")
    json.loads(config.read_text())
    weights = [source / name for name in ["adapter_model.safetensors", "adapter_model.bin"]
               if (source / name).is_file()]
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError(f"Adapter checkpoint has missing or empty weights: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite exported adapter: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".adapter-export-", dir=destination.parent) as temp:
        staging = Path(temp) / "adapter"
        staging.mkdir()
        for path in [config, *weights]:
            shutil.copy2(path, staging / path.name)
        if (source / "README.md").is_file():
            shutil.copy2(source / "README.md", staging / "README.md")
        staging.rename(destination)
    return destination
