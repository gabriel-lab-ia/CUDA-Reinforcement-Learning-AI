from __future__ import annotations

import pytest

from cuda_rl.storage import JsonlDocumentStore


def test_document_store_inserts_and_reads_documents(tmp_path) -> None:
    store = JsonlDocumentStore(tmp_path)

    inserted = store.insert(
        "experiments",
        {"algorithm": "dqn", "reward": 500.0},
        document_id="run-1",
    )

    documents = store.all("experiments")
    assert documents == [inserted]
    assert store.latest("experiments") == inserted


def test_document_store_filters_documents(tmp_path) -> None:
    store = JsonlDocumentStore(tmp_path)
    store.insert("metrics", {"reward": 10.0})
    store.insert("metrics", {"reward": 100.0})

    documents = list(
        store.scan(
            "metrics",
            lambda document: float(document.payload["reward"]) >= 50.0,
        )
    )

    assert len(documents) == 1
    assert documents[0].payload["reward"] == 100.0


def test_document_store_rejects_path_like_collection_names(tmp_path) -> None:
    store = JsonlDocumentStore(tmp_path)

    with pytest.raises(ValueError, match="collection names"):
        store.insert("../escape", {"ok": True})
