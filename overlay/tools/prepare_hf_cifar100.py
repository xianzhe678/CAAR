"""Cache Hugging Face CIFAR-100 once and convert it for offline TOPECL runs."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


REPO_ID = "uoft-cs/cifar100"
FILES = {
    "train": "cifar100/train-00000-of-00001.parquet",
    "test": "cifar100/test-00000-of-00001.parquet",
}


def _metadata_names(path: Path) -> list[str]:
    metadata = pq.read_schema(path).metadata
    if metadata is None or b"huggingface" not in metadata:
        raise ValueError(f"missing Hugging Face schema metadata in {path}")
    features = json.loads(metadata[b"huggingface"])["info"]["features"]
    names = list(features["fine_label"]["names"])
    # The current Hub metadata contains this single truncated class-name typo.
    names = ["crab" if name == "cra" else name for name in names]
    if len(names) != 100 or len(set(names)) != 100:
        raise ValueError("expected 100 unique CIFAR-100 fine-label names")
    return names


def _decode_split(path: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"{path.name} has {parquet.metadata.num_rows} rows, expected {expected_rows}"
        )
    images = np.empty((expected_rows, 32, 32, 3), dtype=np.uint8)
    labels = np.empty(expected_rows, dtype=np.int64)
    offset = 0
    for batch in parquet.iter_batches(
        batch_size=1024, columns=["img", "fine_label"]
    ):
        image_bytes = batch.column("img").field("bytes").to_pylist()
        batch_labels = batch.column("fine_label").to_numpy(zero_copy_only=False)
        for local_index, encoded in enumerate(image_bytes):
            with Image.open(io.BytesIO(encoded)) as image:
                decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if decoded.shape != (32, 32, 3):
                raise ValueError(f"unexpected image shape {decoded.shape}")
            images[offset + local_index] = decoded
        labels[offset : offset + len(batch_labels)] = batch_labels
        offset += len(batch_labels)
    if offset != expected_rows:
        raise RuntimeError(f"decoded {offset} rows, expected {expected_rows}")
    return images, labels


def _validate_labels(labels: np.ndarray, samples_per_class: int, split: str) -> None:
    counts = np.bincount(labels, minlength=100)
    if counts.shape[0] != 100 or not np.all(counts == samples_per_class):
        raise ValueError(f"invalid per-class counts in {split}: {counts.tolist()}")


def prepare(data_root: Path, cache_dir: Path, offline: bool) -> Path:
    output_dir = data_root / "cifar100_hf"
    output_path = output_dir / "cifar100.npz"
    if output_path.is_file():
        print(f"CIFAR-100 offline archive already exists: {output_path}")
        return output_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = {
        split: Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                local_files_only=offline,
            )
        )
        for split, filename in FILES.items()
    }
    names = _metadata_names(parquet_paths["train"])
    train_data, train_targets = _decode_split(parquet_paths["train"], 50000)
    test_data, test_targets = _decode_split(parquet_paths["test"], 10000)
    _validate_labels(train_targets, 500, "train")
    _validate_labels(test_targets, 100, "test")

    output_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix="cifar100_", suffix=".npz", dir=output_dir, delete=False
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        np.savez(
            temporary_path,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets,
            class_names=np.asarray(names),
        )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Prepared offline CIFAR-100 archive: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only files already present in the Hugging Face cache.",
    )
    args = parser.parse_args()
    prepare(args.data_root.resolve(), args.cache_dir.resolve(), args.offline)


if __name__ == "__main__":
    main()

