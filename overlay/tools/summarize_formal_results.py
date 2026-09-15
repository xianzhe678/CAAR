"""Aggregate complete formal manifests as mean ± sample standard deviation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.validate_formal_manifest import HEADS, validate_manifest


METRICS = (
    "final_accuracy",
    "average_stage_accuracy",
    "final_class_average_accuracy",
    "bwf",
    "average_forgetting",
)


def _group_key(manifest: dict) -> tuple:
    run = manifest["run"]
    return (
        run["method"],
        run["semantic_variant"],
        run["dataset"],
        run["backbone"],
        tuple(run["increments"]),
        float(run["fusion_weight"]),
    )


def aggregate(paths: list[Path]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in paths:
        validation = validate_manifest(
            path,
            require_complete=True,
            require_clean_git=True,
            verify_artifacts=False,
        )
        if validation["errors"]:
            raise ValueError(f"invalid formal manifest {path}: {validation['errors']}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        groups[_group_key(manifest)].append(manifest)

    results = []
    for key, manifests in sorted(groups.items()):
        artifact_signatures = {
            json.dumps(
                {
                    name: value.get("sha256", value.get("canonical_sha256"))
                    for name, value in item["run"].get("artifacts", {}).items()
                },
                sort_keys=True,
            )
            for item in manifests
        }
        if len(artifact_signatures) != 1:
            raise ValueError(f"artifact hashes differ across seeds for group {key}")
        seeds = [int(item["run"]["seed"]) for item in manifests]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds in group {key}: {seeds}")
        head_results = {}
        for head in HEADS:
            head_results[head] = {}
            for metric in METRICS:
                values = np.asarray(
                    [item["summary"][head][metric] for item in manifests],
                    dtype=float,
                )
                head_results[head][metric] = {
                    "mean": float(np.around(values.mean(), 2)),
                    "std": float(np.around(values.std(ddof=1), 2)) if len(values) > 1 else 0.0,
                    "values": values.tolist(),
                }
        results.append(
            {
                "method": key[0],
                "variant": key[1],
                "dataset": key[2],
                "backbone": key[3],
                "increments": list(key[4]),
                "fusion_weight": key[5],
                "seeds": sorted(seeds),
                "heads": head_results,
            }
        )
    return results


def markdown_table(results: list[dict]) -> str:
    lines = [
        "| Variant | Head | Final Acc | Avg Stage Acc | BWF | Forgetting | Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in results:
        for head in HEADS:
            metrics = group["heads"][head]
            cell = lambda name: "{:.2f} ± {:.2f}".format(
                metrics[name]["mean"], metrics[name]["std"]
            )
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    group["variant"],
                    head,
                    cell("final_accuracy"),
                    cell("average_stage_accuracy"),
                    cell("bwf"),
                    cell("average_forgetting"),
                    ", ".join(map(str, group["seeds"])),
                )
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    results = aggregate(args.manifests)
    print(markdown_table(results))
    if args.json_output:
        args.json_output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
