# ADR 0001: Local JSONL Document Store

## Status

Accepted

## Context

The project needs a NoSQL-style persistence layer for experiment metadata, but
it should not require an external database for local development, CI, or smoke
tests. Reinforcement-learning experiments produce append-friendly records:
profiles, run lifecycle events, summaries, telemetry snapshots, and benchmark
results.

## Decision

Use append-only JSONL collections as the first document-store backend.

Each collection is stored as:

```text
<root>/<collection>.jsonl
```

Every row is a document with:

- id
- collection
- created_at
- payload

## Consequences

Positive:

- no service dependency
- easy to inspect
- easy to diff
- easy to archive with reports
- compatible with CI
- migration-friendly

Negative:

- no concurrent write coordination
- no query planner
- no secondary indexes
- compaction is manual

## Migration path

The storage interface can later be backed by:

- MongoDB for local or server-side document storage
- DynamoDB for managed cloud metadata
- OpenSearch for searchable experiment logs
- object storage plus manifest indexing

The current JSONL format should remain export-compatible with those systems.
