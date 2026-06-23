from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from cuda_rl.config import ExperimentProfile
from cuda_rl.storage import Document, JsonlDocumentStore, JsonValue

RunStatus = Literal["planned", "running", "completed", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    id: str
    profile_name: str
    algorithm: str
    environment_id: str
    seed: int
    status: RunStatus
    created_at: float
    updated_at: float
    output_directory: str | None = None
    notes: str | None = None

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "profile_name": self.profile_name,
            "algorithm": self.algorithm,
            "environment_id": self.environment_id,
            "seed": self.seed,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "output_directory": self.output_directory,
            "notes": self.notes,
        }


class ExperimentRegistry:
    """Document-store backed registry for experiment lifecycle metadata."""

    def __init__(self, root: Path | str) -> None:
        self.store = JsonlDocumentStore(root)

    def register_profile(self, profile: ExperimentProfile) -> Document:
        return self.store.insert(
            "profiles",
            {
                "run_key": profile.run_key(),
                "profile": _to_json_value(profile.to_dict()),
            },
            document_id=profile.run_key(),
        )

    def start_run(
        self,
        profile: ExperimentProfile,
        *,
        output_directory: Path | str | None = None,
        notes: str | None = None,
    ) -> ExperimentRun:
        now = time.time()
        run = ExperimentRun(
            id=f"{profile.run_key()}-{int(now)}",
            profile_name=profile.name,
            algorithm=profile.algorithm,
            environment_id=profile.environment_id,
            seed=profile.seed,
            status="running",
            created_at=now,
            updated_at=now,
            output_directory=str(output_directory) if output_directory else None,
            notes=notes,
        )
        self.store.insert("runs", run.to_payload(), document_id=run.id)
        return run

    def finish_run(
        self,
        run: ExperimentRun,
        *,
        status: RunStatus,
        notes: str | None = None,
    ) -> ExperimentRun:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(
                "finish_run status must be completed, failed, or cancelled."
            )
        finished = ExperimentRun(
            id=run.id,
            profile_name=run.profile_name,
            algorithm=run.algorithm,
            environment_id=run.environment_id,
            seed=run.seed,
            status=status,
            created_at=run.created_at,
            updated_at=time.time(),
            output_directory=run.output_directory,
            notes=notes or run.notes,
        )
        self.store.insert("runs", finished.to_payload(), document_id=finished.id)
        return finished

    def latest_run(self) -> ExperimentRun | None:
        document = self.store.latest("runs")
        if document is None:
            return None
        return _run_from_document(document)

    def runs_by_status(self, status: RunStatus) -> list[ExperimentRun]:
        return [
            _run_from_document(document)
            for document in self.store.scan(
                "runs",
                lambda item: item.payload.get("status") == status,
            )
        ]


def _run_from_document(document: Document) -> ExperimentRun:
    payload = document.payload
    return ExperimentRun(
        id=document.id,
        profile_name=str(payload["profile_name"]),
        algorithm=str(payload["algorithm"]),
        environment_id=str(payload["environment_id"]),
        seed=_int_payload(payload["seed"], "seed"),
        status=_run_status(str(payload["status"])),
        created_at=_float_payload(payload["created_at"], "created_at"),
        updated_at=_float_payload(payload["updated_at"], "updated_at"),
        output_directory=_optional_str(payload.get("output_directory")),
        notes=_optional_str(payload.get("notes")),
    )


def _run_status(value: str) -> RunStatus:
    if value not in {"planned", "running", "completed", "failed", "cancelled"}:
        raise ValueError(f"invalid run status: {value}")
    return cast(RunStatus, value)


def _optional_str(value: JsonValue | None) -> str | None:
    return None if value is None else str(value)


def _int_payload(value: JsonValue, name: str) -> int:
    if not isinstance(value, str | int | float | bool):
        raise ValueError(f"{name} must be scalar.")
    return int(value)


def _float_payload(value: JsonValue, name: str) -> float:
    if not isinstance(value, str | int | float | bool):
        raise ValueError(f"{name} must be scalar.")
    return float(value)


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    return str(value)
