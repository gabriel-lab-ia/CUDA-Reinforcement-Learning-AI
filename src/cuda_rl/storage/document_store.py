from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    collection: str
    created_at: float
    payload: dict[str, JsonValue]


class JsonlDocumentStore:
    """Tiny append-only NoSQL document store for local experiments."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def insert(
        self,
        collection: str,
        payload: dict[str, JsonValue],
        *,
        document_id: str | None = None,
    ) -> Document:
        self._validate_collection(collection)
        document = Document(
            id=document_id or uuid.uuid4().hex,
            collection=collection,
            created_at=time.time(),
            payload=payload,
        )
        with self._path_for(collection).open("a", encoding="utf-8") as file:
            file.write(json.dumps(self._serialize(document), sort_keys=True))
            file.write("\n")
        return document

    def all(self, collection: str) -> list[Document]:
        return list(self.scan(collection))

    def scan(
        self,
        collection: str,
        predicate: Callable[[Document], bool] | None = None,
    ) -> Iterable[Document]:
        self._validate_collection(collection)
        path = self._path_for(collection)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                document = self._deserialize(json.loads(line))
                if predicate is None or predicate(document):
                    yield document

    def latest(self, collection: str) -> Document | None:
        documents = self.all(collection)
        if not documents:
            return None
        return max(documents, key=lambda document: document.created_at)

    def compact(self, collection: str) -> None:
        """Keep the newest version for each id and rewrite the collection file."""

        self._validate_collection(collection)
        documents_by_id = {document.id: document for document in self.scan(collection)}
        path = self._path_for(collection)
        with path.open("w", encoding="utf-8") as file:
            for document in sorted(
                documents_by_id.values(),
                key=lambda item: item.created_at,
            ):
                file.write(json.dumps(self._serialize(document), sort_keys=True))
                file.write("\n")

    def _path_for(self, collection: str) -> Path:
        return self.root / f"{collection}.jsonl"

    @staticmethod
    def _validate_collection(collection: str) -> None:
        if not collection.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "collection names may contain only letters, numbers, '_' and '-'."
            )

    @staticmethod
    def _serialize(document: Document) -> dict[str, Any]:
        return {
            "id": document.id,
            "collection": document.collection,
            "created_at": document.created_at,
            "payload": document.payload,
        }

    @staticmethod
    def _deserialize(raw: dict[str, Any]) -> Document:
        return Document(
            id=str(raw["id"]),
            collection=str(raw["collection"]),
            created_at=float(raw["created_at"]),
            payload=raw["payload"],
        )
