import json
import tempfile
import unittest
from pathlib import Path

from utils.semantic_prompts import (
    build_candidate_prompts,
    description_bank_sha256,
    descriptions_for_class,
    load_description_bank,
)


class SemanticPromptTests(unittest.TestCase):
    def test_loads_plain_and_wrapped_json(self):
        for payload in (
            {"wolf": ["a gray wolf", "a gray wolf", "pointed ears"]},
            {"classes": {"wolf": ["a gray wolf", "pointed ears"]}},
        ):
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "descriptions.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result = load_description_bank(str(path))
                self.assertEqual(result["wolf"], ["a gray wolf", "pointed ears"])

    def test_normalized_class_lookup_is_conservative(self):
        bank = {"aquarium fish": ["a colorful aquarium fish"]}
        self.assertEqual(
            descriptions_for_class(bank, "aquarium_fish"),
            ["a colorful aquarium fish"],
        )

    def test_missing_description_can_fail_or_use_smoke_fallback(self):
        fallback = build_candidate_prompts("wolf", {}, require_descriptions=False)
        self.assertGreater(len(fallback), 0)
        with self.assertRaises(KeyError):
            build_candidate_prompts("wolf", {}, require_descriptions=True)

    def test_formal_cifar100_bank_is_complete_and_uniform(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "descriptions"
            / "cifar100_visual_attributes_v1.json"
        )
        bank = load_description_bank(str(source))
        self.assertEqual(len(bank), 100)
        self.assertTrue(all(len(prompts) == 8 for prompts in bank.values()))
        self.assertIn("aquarium_fish", bank)
        self.assertIn("willow_tree", bank)
        self.assertEqual(
            description_bank_sha256(source),
            "d907a1bd6edaeafb98eba5f201de436340530ca5a5cd9d3c7828fc89382f9905",
        )


if __name__ == "__main__":
    unittest.main()
