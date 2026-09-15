"""Validate the frozen CIFAR-100 class-level description bank.

This check is intentionally independent of dataset images.  It verifies class
coverage, candidate counts, duplicate prompts, ambiguous-label wording, and
the OpenAI CLIP 77-token limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


CIFAR100_CLASSES = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab",
    "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree",
    "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck",
    "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
    "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)

AMBIGUOUS_LABEL_TERMS = {
    "can": ("metal", "beverage"),
    "mouse": ("mouse animal",),
    "orange": ("orange fruit", "citrus"),
    "plain": ("plain", "landscape"),
    "plate": ("dining plate",),
    "ray": ("ray fish", "stingray"),
    "seal": ("seal animal", "marine seal"),
    "tank": ("military tank", "armored"),
}

CLASS_NAME_ALIASES = {
    "baby": ("human infant",),
    "boy": ("male child",),
    "cattle": ("cow", "bull"),
    "forest": ("woodland",),
    "girl": ("female child",),
    "house": ("home",),
    "lawn_mower": ("lawn mower", "mower"),
    "man": ("adult male",),
    "maple_tree": ("maple",),
    "oak_tree": ("oak",),
    "palm_tree": ("palm",),
    "pickup_truck": ("pickup",),
    "pine_tree": ("pine",),
    "poppy": ("poppies",),
    "sea": ("ocean", "seascape"),
    "sweet_pepper": ("sweet pepper", "bell pepper"),
    "willow_tree": ("willow",),
    "woman": ("adult female",),
}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def validate(path: Path, min_count: int = 8, max_count: int = 12) -> dict:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not isinstance(payload, dict) or not isinstance(payload.get("classes"), dict):
        raise ValueError("JSON must contain an object-valued 'classes' field")
    classes = payload["classes"]

    errors: list[str] = []
    warnings: list[str] = []
    expected = set(CIFAR100_CLASSES)
    actual = set(classes)
    if missing := sorted(expected - actual):
        errors.append(f"missing classes: {missing}")
    if extra := sorted(actual - expected):
        errors.append(f"unexpected classes: {extra}")

    seen: dict[str, str] = {}
    prompts: list[tuple[str, str]] = []
    for class_name in sorted(actual & expected):
        descriptions = classes[class_name]
        if not isinstance(descriptions, list):
            errors.append(f"{class_name}: descriptions must be a list")
            continue
        if not min_count <= len(descriptions) <= max_count:
            errors.append(
                f"{class_name}: expected {min_count}-{max_count} candidates, "
                f"found {len(descriptions)}"
            )
        local_seen: set[str] = set()
        readable_name = class_name.replace("_", " ")
        for index, prompt in enumerate(descriptions):
            label = f"{class_name}[{index}]"
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"{label}: prompt must be a non-empty string")
                continue
            normalized = _normalize(prompt)
            if normalized in local_seen:
                errors.append(f"{label}: duplicate within class")
            local_seen.add(normalized)
            if normalized in seen:
                errors.append(f"{label}: duplicates {seen[normalized]}")
            seen[normalized] = label
            if "\n" in prompt or "\r" in prompt:
                errors.append(f"{label}: prompt must be a single line")
            if not prompt.endswith("."):
                errors.append(f"{label}: prompt must end with a period")
            aliases = (readable_name,) + CLASS_NAME_ALIASES.get(class_name, ())
            if (
                not any(alias in normalized for alias in aliases)
                and class_name not in AMBIGUOUS_LABEL_TERMS
            ):
                warnings.append(f"{label}: class name is not explicit")
            prompts.append((label, prompt))

        required = AMBIGUOUS_LABEL_TERMS.get(class_name)
        if required:
            for index, prompt in enumerate(descriptions):
                if not any(term in _normalize(prompt) for term in required):
                    errors.append(
                        f"{class_name}[{index}]: ambiguous label must contain one of {required}"
                    )

    token_backend = "unavailable"
    max_tokens = None
    try:
        import clip

        token_backend = "openai-clip"
        token_lengths = []
        for label, prompt in prompts:
            try:
                tokens = clip.tokenize([prompt], truncate=False)
            except RuntimeError as exc:
                errors.append(f"{label}: exceeds CLIP context length ({exc})")
                continue
            length = int((tokens[0] != 0).sum().item())
            token_lengths.append(length)
        max_tokens = max(token_lengths, default=0)
    except ImportError:
        warnings.append("OpenAI clip is unavailable; exact 77-token validation skipped")

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "class_count": len(classes),
        "prompt_count": len(prompts),
        "tokenizer": token_backend,
        "max_clip_tokens": max_tokens,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("descriptions/cifar100_visual_attributes_v1.json"),
    )
    args = parser.parse_args()
    result = validate(args.path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
