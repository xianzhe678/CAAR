"""Build the frozen ImageNet-R text bank source from the public CuPL prompts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


IMAGENET_R_INDICES = (
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76,
    79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132,
    144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178,
    187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247,
    250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289,
    291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323,
    327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362,
    365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425,
    428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472,
    476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596,
    609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774,
    776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875,
    883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947,
    948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988,
)


def load_cupl_class_names(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    _, expression = source.split("=", 1)
    names = ast.literal_eval(expression.strip())
    if not isinstance(names, list) or len(names) != 1000:
        raise ValueError("CuPL ImageNet class-name file must contain 1000 names")
    return names


def build(args: argparse.Namespace) -> None:
    class_index = json.loads(args.imagenet_index.read_text(encoding="utf-8"))
    cupl_names = load_cupl_class_names(args.cupl_classes)
    cupl_prompts = json.loads(args.cupl_prompts.read_text(encoding="utf-8"))

    entries = []
    descriptions = {}
    for index in IMAGENET_R_INDICES:
        wnid, official_name = class_index[str(index)]
        cupl_name = cupl_names[index]
        candidates = list(dict.fromkeys(
            prompt.strip() for prompt in cupl_prompts[cupl_name] if prompt.strip()
        ))
        if len(candidates) < args.candidates_per_class:
            raise ValueError(f"{cupl_name!r} has only {len(candidates)} CuPL prompts")
        entries.append({
            "imagenet_index": index,
            "wnid": wnid,
            "class_name": official_name,
            "cupl_class_name": cupl_name,
        })
        descriptions[official_name] = candidates[: args.candidates_per_class]

    if len(entries) != 200 or [item["wnid"] for item in entries] != sorted(item["wnid"] for item in entries):
        raise ValueError("ImageNet-R mapping must contain 200 classes in loader order")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "metadata": {
            "dataset": "ImageNet-R",
            "version": "1.0",
            "scope": "frozen class-level descriptions",
            "image_access": "none",
            "source": "CuPL_image_prompts.json",
            "candidate_count_per_class": args.candidates_per_class,
            "intended_use": "select four frozen prompts per class with CLIP-based MMR",
        },
        "classes": descriptions,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_output.write_text(json.dumps({
        "metadata": {"dataset": "ImageNet-R", "class_count": 200},
        "classes": entries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cupl-prompts", type=Path, required=True)
    parser.add_argument("--cupl-classes", type=Path, required=True)
    parser.add_argument("--imagenet-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--candidates-per-class", type=int, default=8)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
