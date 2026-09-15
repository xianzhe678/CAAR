"""Validate a formal TOPECL metrics manifest before using it in a paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.semantic_prompts import description_bank_sha256


HEADS = ("text", "orth", "fusion")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(
    path: Path,
    require_complete: bool = False,
    require_clean_git: bool = False,
    verify_artifacts: bool = True,
) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("unsupported or missing schema_version")
    run = manifest.get("run")
    stages = manifest.get("stages")
    summary = manifest.get("summary")
    if not isinstance(run, dict) or not isinstance(stages, list) or not isinstance(summary, dict):
        errors.append("manifest must contain run, stages, and summary")
        return {"path": str(path.resolve()), "errors": errors, "warnings": warnings}

    task_count = int(run.get("task_count", 0))
    task_ids = [stage.get("task_id") for stage in stages]
    if task_ids != list(range(len(stages))):
        errors.append(f"stage ids are not consecutive: {task_ids}")
    if require_complete and (not manifest.get("complete") or len(stages) != task_count):
        errors.append("run is incomplete")
    git = run.get("git", {})
    if git.get("dirty") is not False:
        message = "run does not have a verified clean git work tree"
        (errors if require_clean_git else warnings).append(message)
    if require_clean_git and not git.get("commit"):
        errors.append("run does not record a git commit")
    configuration = run.get("configuration")
    if isinstance(configuration, dict):
        canonical = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != run.get("configuration_sha256"):
            errors.append("configuration hash mismatch")
    else:
        warnings.append("run does not contain the expanded configuration")

    evaluation_source = run.get("evaluation_source", "test")
    if evaluation_source not in {"test", "validation"}:
        errors.append(f"unsupported evaluation_source: {evaluation_source}")
    if evaluation_source == "validation":
        split = run.get("validation_split")
        if not isinstance(split, dict):
            errors.append("validation run is missing validation_split provenance")
        elif not split.get("validation_indices_sha256"):
            errors.append("validation split is missing its index hash")
    if (
        run.get("run_scope") == "diagnostic_prefix"
        and isinstance(configuration, dict)
        and configuration.get("formal_require_validation_for_screening")
        and evaluation_source != "validation"
    ):
        errors.append("screening protocol touched test instead of validation")

    for stage in stages:
        task_id = stage["task_id"]
        for head in HEADS:
            metrics = stage.get("heads", {}).get(head)
            if not isinstance(metrics, dict):
                errors.append(f"stage {task_id}: missing {head} metrics")
                continue
            expected_tasks = task_id + 1
            if len(metrics.get("per_task_accuracy", [])) != expected_tasks:
                errors.append(f"stage {task_id} {head}: wrong per-task metric count")
        if stage.get("evaluation_source", evaluation_source) != evaluation_source:
            errors.append(f"stage {task_id}: evaluation source mismatch")
        geometry = stage.get("geometry", {})
        if geometry.get("type") == "stable_semantic_codebook":
            if geometry.get("old_code_drift", 1.0) >= 1e-6:
                errors.append(f"stage {task_id}: old code drift invariant failed")
            if geometry.get("normalized_gram_error", 1.0) >= 1e-5:
                errors.append(f"stage {task_id}: Gram invariant failed")

    for head in HEADS:
        values = summary.get(head)
        if stages and not isinstance(values, dict):
            errors.append(f"summary is missing {head}")
            continue
        if not stages:
            continue
        if len(values.get("stage_accuracy_curve", [])) != len(stages):
            errors.append(f"summary {head}: stage curve length mismatch")
        if values.get("final_accuracy") != stages[-1]["heads"][head]["overall_accuracy"]:
            errors.append(f"summary {head}: final accuracy does not match last stage")

    if verify_artifacts:
        for name, artifact in run.get("artifacts", {}).items():
            artifact_path = Path(artifact.get("path", ""))
            if not artifact_path.is_file():
                warnings.append(f"artifact is unavailable on this machine: {name}")
                continue
            if "canonical_sha256" in artifact:
                actual = description_bank_sha256(artifact_path)
                expected = artifact["canonical_sha256"]
            else:
                actual = sha256_file(artifact_path)
                expected = artifact.get("sha256")
            if actual != expected:
                errors.append(f"artifact hash mismatch: {name}")
        for stage in stages:
            checkpoint = stage.get("checkpoint")
            if not checkpoint:
                continue
            checkpoint_path = Path(checkpoint["path"])
            if not checkpoint_path.is_file():
                warnings.append(f"checkpoint is unavailable: task {stage['task_id']}")
            elif sha256_file(checkpoint_path) != checkpoint["sha256"]:
                errors.append(f"checkpoint hash mismatch: task {stage['task_id']}")

    return {
        "path": str(path.resolve()),
        "seed": run.get("seed"),
        "variant": run.get("semantic_variant"),
        "stage_count": len(stages),
        "complete": bool(manifest.get("complete")),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--skip-artifact-hashes", action="store_true")
    args = parser.parse_args()
    result = validate_manifest(
        args.manifest,
        require_complete=args.require_complete,
        require_clean_git=args.require_clean_git,
        verify_artifacts=not args.skip_artifact_hashes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
