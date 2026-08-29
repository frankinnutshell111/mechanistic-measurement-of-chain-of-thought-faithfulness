"""Append-only JSONL checkpoints with duplicate-completion protection."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class DuplicateCompletionError(RuntimeError):
    """Raised before a completed question would be written a second time."""


class JsonlCheckpoint:
    def __init__(
        self,
        path: str | Path,
        *,
        resume: bool = True,
        key_field: str = "question_id",
    ) -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed: set[str] = set()
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            if not resume and self.path.stat().st_size:
                raise FileExistsError(
                    f"Refusing to overwrite existing checkpoint without --resume: {self.path}"
                )
            self._load_completed()

    def _load_completed(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record_key = str(record[self.key_field])
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid checkpoint record at {self.path}:{line_number}"
                    ) from exc
                if record.get("terminal", True):
                    if record_key in self._completed:
                        raise ValueError(
                            f"Duplicate completed key {record_key!r} at {self.path}:{line_number}"
                        )
                    self._completed.add(record_key)
                    self._records[record_key] = record

    @property
    def completed_ids(self) -> frozenset[str]:
        return frozenset(self._completed)

    @property
    def completed_records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._records.values())

    def is_completed(self, record_key: str) -> bool:
        return record_key in self._completed

    def append_terminal(self, record: Mapping[str, Any]) -> None:
        record_key = str(record[self.key_field])
        if record_key in self._completed:
            raise DuplicateCompletionError(
                f"Record {self.key_field}={record_key!r} is already complete"
            )
        payload = dict(record)
        payload["terminal"] = True
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._completed.add(record_key)
        self._records[record_key] = payload
