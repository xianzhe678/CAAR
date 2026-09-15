"""Download and validate the OpenAI CLIP checkpoint used by TOPECL."""

from __future__ import annotations

import argparse
from pathlib import Path

import clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model, _ = clip.load(
        "ViT-B/16",
        device="cpu",
        jit=False,
        download_root=str(args.cache_dir.resolve()),
    )
    print(
        "OpenAI CLIP cache ready:",
        args.cache_dir.resolve() / "ViT-B-16.pt",
        "feature_dim=",
        int(model.text_projection.shape[1]),
    )


if __name__ == "__main__":
    main()

