# DocWorkspace Developer Guide

DocWorkspace is the Python package that models workspace data as a graph of
Polars LazyFrame nodes.

- [Architecture](architecture.md): big-picture package role.
- [Node and workspace model](node-and-workspace.md): node proxy behavior,
  graph relationships, document columns, and tokenization metadata.
- [Persistence](persistence.md): `metadata.json`, `.plbin` files, path
  rebasing, and workspace data garbage collection.
- [Testing and release](testing.md): local validation and release notes.
