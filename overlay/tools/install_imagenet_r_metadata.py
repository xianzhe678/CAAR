"""Install the class-name metadata expected by the TOPECL ImageNet-R loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    entries = payload["classes"]
    expected = [entry["wnid"] for entry in entries]
    train = sorted(path.name for path in (args.dataset_root / "train").iterdir() if path.is_dir())
    test = sorted(path.name for path in (args.dataset_root / "test").iterdir() if path.is_dir())
    if train != expected or test != expected:
        raise ValueError("ImageNet-R folders do not match the locked 200-class mapping")

    header = [f"# ImageNet-R class mapping {index + 1}" for index in range(13)]
    rows = [f'{entry["wnid"]} {entry["class_name"]}' for entry in entries]
    (args.dataset_root / "README.txt").write_text(
        "\n".join(header + rows) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
