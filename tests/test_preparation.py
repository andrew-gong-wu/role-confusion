import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_probe.preparation import (  # noqa: E402
    ROLES,
    build_role_variants,
    render_single_role_gptoss,
)


class PreparationTests(unittest.TestCase):
    def test_all_roles_preserve_content(self):
        content = "Beginners BBQ Class!"
        rows = build_role_variants([content])

        self.assertEqual([row["role"] for row in rows], list(ROLES))
        self.assertEqual({row["content"] for row in rows}, {content})
        self.assertEqual({row["base_seq_ix"] for row in rows}, {0})
        self.assertEqual(len({row["prompt_ix"] for row in rows}), len(ROLES))
        for row in rows:
            self.assertIn(content, row["prompt"])

    def test_expected_harmony_headers(self):
        self.assertEqual(
            render_single_role_gptoss("user", "hello"),
            "<|start|>user<|message|>hello<|end|>",
        )
        self.assertEqual(
            render_single_role_gptoss("cot", "hello"),
            "<|start|>assistant<|channel|>analysis<|message|>hello<|end|>",
        )
        self.assertEqual(
            render_single_role_gptoss("assistant", "hello"),
            "<|start|>assistant<|channel|>final<|message|>hello<|end|>",
        )

    def test_base_sequence_groups_role_copies(self):
        rows = build_role_variants(["first", "second"])
        grouped = {
            base_ix: [row for row in rows if row["base_seq_ix"] == base_ix]
            for base_ix in {row["base_seq_ix"] for row in rows}
        }
        self.assertEqual(set(grouped), {0, 1})
        self.assertTrue(all(len(group) == len(ROLES) for group in grouped.values()))
        self.assertEqual({row["content"] for row in grouped[0]}, {"first"})
        self.assertEqual({row["content"] for row in grouped[1]}, {"second"})

    def test_invalid_role_fails(self):
        with self.assertRaises(ValueError):
            render_single_role_gptoss("banana", "hello")


if __name__ == "__main__":
    unittest.main()

