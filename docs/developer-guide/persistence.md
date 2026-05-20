# Persistence

## Workspace Folder

A saved workspace is a folder containing:

- `metadata.json`,
- `data/{node_id}.plbin` files,
- data files referenced by lazy scan plans,
- optional dotfile artifacts owned by analysis caches.

`Workspace.save(path)` delegates to `workspace.io.write_workspace()`.
`Workspace.load(path)` delegates to `workspace.io.read_workspace()`.

## Node Serialization

`node.io.to_dict()` writes the node's LazyFrame plan with:

```python
node.data.serialize(abs_data_path, format="binary")
```

The returned JSON metadata stores node id, name, operation, document column,
derived metadata, parent ids, and a workspace-relative `data_path`.

`node.io.from_dict()` deserializes the `.plbin` file with Polars and recreates
the node. If a workspace is provided, parent ids that already exist in the
workspace are resolved to node objects.

## Workspace Metadata

`workspace.io.write_workspace()` writes:

- workspace id,
- name,
- version,
- description,
- created/modified timestamps,
- serialized node entries.

The current workspace version is `2`.

## Path Rebasing

Polars serialized plans can contain absolute source paths. Moving a workspace
folder between directories or machines can break those paths.

`rebase_workspace_sources(path)` fixes this before load:

1. read `metadata.json`,
2. find each registered `.plbin`,
3. ask `polars_text.list_source_paths()` for embedded scan paths,
4. map old paths to files with the same basename in the current `data/` folder,
5. call `polars_text.replace_source_paths()` to rewrite the plan in place.

This avoids collecting or rewriting the underlying data.

## Save-Time Garbage Collection

After writing metadata, `write_workspace()` removes unreferenced data files:

- `.plbin` files whose names are not registered in metadata,
- non-dotfile `.parquet` files that are not referenced by any registered plan.

Dotfile parquet files are skipped. They are treated as out-of-band analysis
artifacts whose lifecycle is owned by the backend analysis cache, not the
workspace serializer.
