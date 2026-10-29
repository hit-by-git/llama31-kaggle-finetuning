import json
import tempfile
import unittest
from pathlib import Path
from llama31_finetune.checkpoints import export_adapter


class CheckpointTests(unittest.TestCase):
    def make_checkpoint(self, root):
        checkpoint = root / "checkpoint-10"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
        (checkpoint / "adapter_model.safetensors").write_bytes(b"test-adapter-payload")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer-stays-in-checkpoint")
        return checkpoint

    def test_export_preserves_adapter_bytes_and_leaves_optimizer_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            result = export_adapter(checkpoint, root / "adapter")
            self.assertEqual((result / "adapter_model.safetensors").read_bytes(), b"test-adapter-payload")
            self.assertFalse((result / "optimizer.pt").exists())
            self.assertTrue((checkpoint / "optimizer.pt").exists())

    def test_incomplete_checkpoint_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            (checkpoint / "adapter_model.safetensors").write_bytes(b"")
            with self.assertRaises(ValueError):
                export_adapter(checkpoint, root / "adapter")
            self.assertFalse((root / "adapter").exists())

    def test_existing_export_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            export_adapter(checkpoint, root / "adapter")
            with self.assertRaises(FileExistsError):
                export_adapter(checkpoint, root / "adapter")


if __name__ == "__main__":
    unittest.main()
