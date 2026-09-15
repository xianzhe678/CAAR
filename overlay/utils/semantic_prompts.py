"""Description-bank loading and prompt construction for semantic TOPECL."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_VISUAL_TEMPLATES = (
    "a close-up photo of a {}.",
    "a photo showing the shape of a {}.",
    "a photo showing the color and texture of a {}.",
    "a side-view photo of a {}.",
    "a {} in its typical environment.",
    "a photo showing the distinctive visual features of a {}.",
)


def _normalized_name(name: str) -> str:
    return " ".join(name.replace("_", " ").replace("-", " ").lower().split())


def load_description_bank(path: str | None) -> dict[str, list[str]]:
    """Load ``{class_name: [description, ...]}`` from JSON.

    A top-level ``classes`` field is also accepted so the JSON may contain
    metadata alongside the actual mapping.
    """
    if path is None or not str(path).strip():
        return {}
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"semantic description file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("description JSON must be an object")
    if "classes" in payload:
        payload = payload["classes"]
    if not isinstance(payload, dict):
        raise ValueError("description JSON 'classes' must be an object")

    result: dict[str, list[str]] = {}
    for class_name, descriptions in payload.items():
        if not isinstance(class_name, str) or not isinstance(descriptions, list):
            raise ValueError("each description entry must map a class string to a list")
        if not all(isinstance(item, str) for item in descriptions):
            raise ValueError(f"all descriptions for '{class_name}' must be strings")
        cleaned = [item.strip() for item in descriptions if item.strip()]
        result[class_name] = list(dict.fromkeys(cleaned))
    return result


def description_bank_sha256(path: str | Path) -> str:
    """Hash semantic JSON content independently of whitespace/line endings."""
    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def descriptions_for_class(
    description_bank: Mapping[str, Sequence[str]], class_name: str
) -> list[str]:
    """Look up a class exactly first, then by a conservative normalized name."""
    if class_name in description_bank:
        return list(description_bank[class_name])
    target = _normalized_name(class_name)
    matches = [
        list(value)
        for key, value in description_bank.items()
        if _normalized_name(key) == target
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous normalized description key for '{class_name}'")
    return matches[0] if matches else []


def build_candidate_prompts(
    class_name: str,
    description_bank: Mapping[str, Sequence[str]],
    require_descriptions: bool = False,
) -> list[str]:
    """Return class-specific full prompts or deterministic smoke-test prompts."""
    prompts = descriptions_for_class(description_bank, class_name)
    if prompts:
        return list(dict.fromkeys(prompts))
    if require_descriptions:
        raise KeyError(f"no semantic descriptions found for class '{class_name}'")
    return [template.format(class_name) for template in DEFAULT_VISUAL_TEMPLATES]
