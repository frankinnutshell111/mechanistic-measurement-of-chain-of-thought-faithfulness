"""Small deterministic helpers shared across experiment phases."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a platform-stable 63-bit seed from explicit experiment identifiers."""

    payload = json.dumps([base_seed, *map(str, parts)], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def first_present_version(distributions: Iterable[str]) -> str | None:
    for name in distributions:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def runtime_metadata(model: Any | None = None) -> dict[str, Any]:
    model_revision = None
    if model is not None:
        model_config = getattr(model, "config", None)
        model_revision = getattr(model_config, "_commit_hash", None)
    gpu_name = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "model_revision": model_revision,
        "torch_version": first_present_version(("torch",)),
        "transformers_version": first_present_version(("transformers",)),
        "datasets_version": first_present_version(("datasets",)),
        "gpu_name": gpu_name,
    }


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_write_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
