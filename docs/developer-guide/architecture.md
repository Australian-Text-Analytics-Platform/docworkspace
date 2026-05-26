# DocWorkspace Architecture

DocWorkspace is a small Python library that wraps Polars `LazyFrame` objects in
a directed node graph. In this monorepo, the backend uses it as the workspace
execution-lineage model.

## Package Role

DocWorkspace does not implement the web API, user files, authentication, or
analysis feature logic. Its job is narrower:

- hold nodes backed by lazy Polars plans,
- track parent/child relationships between transformations,
- preserve document-column and tokenization metadata,
- serialize workspaces to portable folders,
- expose simple JSON summaries and graph payloads.

## Main Modules

- `src/docworkspace/node/core.py`: `Node`, tokenization metadata, Polars
  delegation, explicit dataframe operations, undo/redo, and schema helpers.
- `src/docworkspace/node/io.py`: node serialization to JSON metadata plus
  `.plbin` LazyFrame plan files.
- `src/docworkspace/workspace/core.py`: `Workspace`, node registry, parent
  resolution, add/remove behavior, and persistence shims.
- `src/docworkspace/workspace/io.py`: workspace read/write, path rebasing, and
  data-dir garbage collection.
- `src/docworkspace/workspace/analysis.py`: lightweight JSON summaries and graph
  payloads for API/UI use.

## Design Principle

The package keeps the core deliberately small. Advanced graph algorithms were
removed from the core because current backend usage needs only node lookup,
root/leaf discovery, parent/child traversal, add/remove, and persistence.
Reintroduce broader graph algorithms only with real call sites and tests.
