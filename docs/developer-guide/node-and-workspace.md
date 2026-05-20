# Node And Workspace Model

## Node

`Node` wraps a Polars `LazyFrame`. The constructor requires lazy data; eager
`DataFrame` objects should be converted with `.lazy()` before creating a node.

Each node stores:

- `id` and `name`,
- `data`, the wrapped `LazyFrame`,
- `parents`, as node objects or unresolved string ids,
- `workspace`, if attached,
- `operation`, describing how it was produced,
- `document`, the text/document column name,
- `derived`, metadata for hidden generated columns.

## Polars Delegation

`Node.__getattr__` delegates unknown attributes to the underlying LazyFrame.
If the delegated call returns another LazyFrame, the result is wrapped in a new
child node with the current node as parent. If the call returns a schema,
scalar, or other non-LazyFrame value, it is returned directly.

Critical operations are explicit methods instead of pure delegation:

- `filter()` creates a child node and preserves metadata.
- `select()` creates a child node and prunes derived metadata to selected
  columns.
- `join()` creates a two-parent child node and rejects conflicting derived
  metadata for retained generated columns.
- `slice()` creates a child node.
- `drop()` creates a child node and drops derived columns whose source user
  columns were removed.
- `rename()` mutates the current node in place, maps the document column, and
  invalidates derived columns whose source names changed.

## Undo And Redo

Changing `node.data` pushes the previous LazyFrame onto an undo stack and
clears the redo stack. Undo/redo only tracks in-place data replacement on that
node; graph-producing operations create child nodes instead. The undo stack is
capped by `MAX_UNDO_DEPTH`.

## Derived Columns

Derived analytical columns live in the same LazyFrame as user columns but are
tracked in `Node.derived`. Each entry records the source user column, form,
model, language, generated timestamp, and optional cache filename.

Derived metadata must stay synchronized with the LazyFrame schema. Selecting
or dropping columns prunes stale metadata. Renaming a source column invalidates
derived columns because their source reference no longer matches the data.

## Workspace

`Workspace` is a node registry with metadata. It stores nodes by id and resolves
string parent references to node objects when possible.

Adding a node:

- moves it out of a previous workspace if needed,
- attaches it to the new workspace,
- resolves its parents,
- updates existing children that were waiting on that parent id.

Removing a node rewires its children to inherit the removed node's parents,
then deletes the node and its `.plbin` file if present. This keeps downstream
branches reachable without requiring a full graph rebuild.
