"""CPU-only contracts for prompts, completion targets and experiment sampling."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llama31_finetune.config import experiment_dir, load_config, run_groups
from llama31_finetune.prompting import encode_row, label_maps, make_prompt
from llama31_finetune.sampling import combine_tasks


class FakeChatTokenizer:
    """Character-level chat format with one explicit end-of-turn token."""
    end_turn = 2_000_000

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token - 1) for token in tokens if token != self.end_turn)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for message in messages:
            ids.extend(self.encode(f"<{message['role']}>\n{message['content']}"))
            ids.append(self.end_turn)
        if add_generation_prompt:
            ids.extend(self.encode("<assistant>\n"))
        return ids


class BrokenPrefixTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        if messages[-1]["role"] == "assistant":
            ids[0] = 0
        return ids


def example(task="misinfo"):
    rows = {
        "misinfo": {"article": "The court met on Wednesday.",
                    "post": "The court met on Monday.", "label": "misinfo"},
        "hasoc": {"text": "A sample tweet.", "label": "HOF"},
        "constraint": {"tweet": "A COVID-19 sample claim.", "label": "fake"},
    }
    return {"task": task, **rows[task]}


class PromptingTests(unittest.TestCase):
    def setUp(self):
        self.tok = FakeChatTokenizer()

    def test_only_answer_and_turn_end_are_supervised(self):
        row = example()
        _, prefix, _ = make_prompt(row, self.tok, 1024)
        encoded, truncated = encode_row(row, self.tok, 1024)
        self.assertFalse(truncated)
        self.assertEqual(encoded["labels"][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(encoded["labels"][len(prefix):], [ord("B") + 1, self.tok.end_turn])
        self.assertEqual(encoded["input_ids"][len(prefix):], encoded["labels"][len(prefix):])
        self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))

    def test_long_input_truncates_fields_without_losing_answer(self):
        row = example()
        row["article"] = "reference information " * 500
        row["post"] = "post information " * 500
        encoded, truncated = encode_row(row, self.tok, 512)
        messages, _, prompt_truncated = make_prompt(row, self.tok, 512)
        self.assertTrue(truncated)
        self.assertTrue(prompt_truncated)
        self.assertLessEqual(len(encoded["input_ids"]), 512)
        supervised = [value for value in encoded["labels"] if value != -100]
        self.assertEqual(self.tok.decode(supervised), "B")
        self.assertEqual(supervised[-1], self.tok.end_turn)
        self.assertEqual(set(json.loads(messages[1]["content"])), {"article", "post"})

    def test_leakage_metadata_never_changes_prompt(self):
        row = example()
        polluted = {**row, "analysis": "Answer is A", "variant": "misinfo",
                    "false_claim": "secret target", "tactic": "wrong_date",
                    "contradicting_fact": "hidden annotation", "event_id": 123,
                    "language": "plain English", "tone": -8}
        self.assertEqual(make_prompt(row, self.tok, 1024),
                         make_prompt(polluted, self.tok, 1024))

    def test_label_maps_and_input_fields_are_task_specific(self):
        specs = {
            "misinfo": (("not_misinfo", "misinfo"), {"article", "post"}),
            "hasoc": (("NOT", "HOF"), {"text"}),
            "constraint": (("real", "fake"), {"tweet"}),
        }
        for task, (labels, fields) in specs.items():
            with self.subTest(task=task):
                forward, reverse = label_maps(task)
                self.assertEqual(forward, dict(zip(labels, ("A", "B"))))
                self.assertEqual(reverse, {"A": labels[0], "B": labels[1]})
                messages, _, _ = make_prompt(example(task), self.tok, 1024)
                self.assertEqual(set(json.loads(messages[1]["content"])), fields)
                for label, code in zip(labels, ("A", "B")):
                    encoded, _ = encode_row({**example(task), "label": label}, self.tok, 1024)
                    self.assertEqual(self.tok.decode([v for v in encoded["labels"] if v != -100]), code)

    def test_prefix_mismatch_fails_instead_of_masking_wrong_tokens(self):
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            encode_row(example(), BrokenPrefixTokenizer(), 1024)

    def test_impossible_prompt_budget_fails(self):
        with self.assertRaisesRegex(ValueError, "headers exceed"):
            encode_row(example(), self.tok, 20)

    def test_wrong_task_label_is_rejected(self):
        with self.assertRaises(KeyError):
            encode_row({**example("hasoc"), "label": "fake"}, self.tok, 1024)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.frames = {
            task: pd.DataFrame({"task": [task] * n, "label": ["A", "B"] * (n // 2),
                                "id": [f"{task}-{i}" for i in range(n)]})
            for task, n in [("misinfo", 8), ("hasoc", 20), ("constraint", 40)]
        }

    def test_balanced_counts_and_repeatability(self):
        combined = combine_tasks(self.frames, seed=42, balanced=True)
        self.assertEqual(combined.task.value_counts().to_dict(), {task: 40 for task in self.frames})
        pd.testing.assert_frame_equal(combined, combine_tasks(self.frames, seed=42, balanced=True))
        self.assertFalse(combined.equals(combine_tasks(self.frames, seed=43, balanced=True)))

    def test_unbalanced_sampling_preserves_all_original_examples(self):
        combined = combine_tasks(self.frames, seed=42, balanced=False)
        self.assertEqual(len(combined), sum(len(frame) for frame in self.frames.values()))
        self.assertEqual(set(combined.id), set(pd.concat(self.frames.values()).id))

    def test_smoke_preserves_both_classes_per_task(self):
        for balanced in (False, True):
            for seed in (0, 1, 42, 123, 2026):
                with self.subTest(balanced=balanced, seed=seed):
                    combined = combine_tasks(self.frames, seed=seed, balanced=balanced, smoke=True)
                    for task, frame in combined.groupby("task"):
                        self.assertEqual(set(frame.label), {"A", "B"}, task)
                    if balanced:
                        self.assertEqual(set(combined.task.value_counts()), {32})


class ConfigTests(unittest.TestCase):
    def test_smoke_full_and_task_outputs_are_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            base = {"data_root": directory, "output_dir": str(Path(directory) / "runs")}
            path.write_text(json.dumps({**base, "smoke_test": True}))
            smoke = load_config(path)
            path.write_text(json.dumps({**base, "smoke_test": False}))
            full = load_config(path)
            self.assertNotEqual(experiment_dir(smoke, "misinfo"), experiment_dir(full, "misinfo"))
            self.assertNotEqual(experiment_dir(full, "misinfo"), experiment_dir(full, "hasoc"))
            self.assertEqual(experiment_dir(smoke, "misinfo").parts[-2:], ("smoke", "misinfo"))

    def test_training_groups_do_not_mix_modes(self):
        self.assertEqual(run_groups({"mode": "separate", "tasks": ["misinfo", "hasoc"]}),
                         [("misinfo", ["misinfo"]), ("hasoc", ["hasoc"])])
        self.assertEqual(run_groups({"mode": "multitask", "tasks": ["misinfo", "hasoc"]}),
                         [("multitask", ["misinfo", "hasoc"])])


if __name__ == "__main__":
    unittest.main()
