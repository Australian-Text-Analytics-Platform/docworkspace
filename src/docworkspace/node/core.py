"""Node core definition (split from former monolithic node.py).

Contains structural aspects: construction, parent/child tracking, schema helpers,
and core dataframe operations (join/filter/slice/dynamic delegation).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    NotRequired,
    Sequence,
    TypedDict,
    cast,
)

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from ..workspace.core import Workspace  # pragma: no cover


class DerivedColumnMeta(TypedDict):
    """Metadata for a derived analytic spec (Phase 2, decision 7).

    ``source_column`` points at the originating user column; ``form`` says what
    kind of derivation (``"tokens"``, future: ``"pos"``, ``"ner"``); ``model``
    identifies the backend that produced it. The physical derived column may be
    stored in the LazyFrame for legacy workspaces or hydrated temporarily from a
    cache backend for current token specs.
    """

    source_column: str
    form: str
    model: str
    language: str | None
    generated_at: str
    cache_filename: NotRequired[str]
    cache_backend: NotRequired[str]
    cache_schema_version: NotRequired[int]
    params: NotRequired[dict[str, Any]]


class Node:
    MAX_UNDO_DEPTH = 50

    @staticmethod
    def _lazyframe_height(data: pl.LazyFrame) -> int:
        collected = cast(pl.DataFrame, data.select(pl.len()).collect())
        return int(collected.item())

    def __init__(
        self,
        data: pl.LazyFrame,
        name: str,
        workspace: Workspace | None = None,
        parents: Sequence["Node | str"] = (),
        operation: str | None = None,
        id: str | None = None,
        document: str | None = None,
        derived: Mapping[str, DerivedColumnMeta] | None = None,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.name = name or f"node_{self.id[:8]}"

        if not isinstance(data, pl.LazyFrame):
            raise TypeError(
                "Node data must be a polars LazyFrame "
                f"(received {type(data).__name__})."
            )
        self._undo_stack: list[pl.LazyFrame] = []
        self._redo_stack: list[pl.LazyFrame] = []
        self._data: pl.LazyFrame = data
        self._document_column: str | None = document
        # Per-column metadata for derived analytic specs. Keys are stable
        # derived column names (e.g. "__derived__.tokens.text.jieba"); values
        # carry source_column / form / model / language / generated_at and
        # cache details. The physical column may be hydrated only temporarily.
        # Empty dict on legacy nodes is fully backward compatible.
        self.derived = cast(
            dict[str, DerivedColumnMeta],
            {k: dict(v) for k, v in derived.items()} if derived else {},
        )
        self.parents: list[Node | str] = list(parents)
        self.workspace: Workspace | None = workspace
        self.operation = operation

        if self.workspace is not None and self.id not in self.workspace.nodes:
            self.workspace.add_node(self)

    @staticmethod
    def _parent_id(parent: "Node | str") -> str:
        return parent.id if isinstance(parent, Node) else str(parent)

    @staticmethod
    def _parent_matches(parent: "Node | str", node: "Node") -> bool:
        return parent is node if isinstance(parent, Node) else str(parent) == node.id

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - thin wrapper
        # Delegate attribute access to underlying data object. Callable
        # LazyFrame results are wrapped as graph child nodes; scalar/schema
        # results are returned unchanged.
        attr = getattr(self.data, item)
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if not isinstance(result, pl.LazyFrame):
                    return result

                return self._child_node(
                    data=result,
                    name=f"{item}_{self.name}",
                    operation=item,
                    document=self.document,
                )

            return wrapper
        return attr

    def _child_node(
        self,
        *,
        data: pl.LazyFrame,
        name: str,
        operation: str,
        parents: Sequence["Node | str"] = (),
        derived: Mapping[str, DerivedColumnMeta] | None = None,
        document: str | None = None,
    ) -> "Node":
        child = Node(
            data=data,
            name=name,
            workspace=self.workspace,
            parents=parents or [self],
            operation=operation,
            derived=self.derived if derived is None else derived,
        )
        if document is not None:
            child.document = document
        return child

    @staticmethod
    def _derived_for_columns(
        derived: Mapping[str, DerivedColumnMeta], columns: set[str]
    ) -> dict[str, DerivedColumnMeta]:
        return {
            name: meta
            for name, meta in derived.items()
            if name in columns or meta.get("source_column") in columns
        }

    @classmethod
    def _drop_derived_from_stale_sources(
        cls,
        data: pl.LazyFrame,
        derived: Mapping[str, DerivedColumnMeta],
        stale_sources: set[str],
    ) -> tuple[pl.LazyFrame, dict[str, DerivedColumnMeta]]:
        current_columns = set(data.collect_schema().names())
        retained = cls._derived_for_columns(derived, current_columns)
        if not stale_sources:
            return data, retained

        cascade_targets = [
            name
            for name, meta in retained.items()
            if meta["source_column"] in stale_sources
        ]
        for name in cascade_targets:
            retained.pop(name, None)

        if cascade_targets:
            data = data.drop(*cascade_targets, strict=False)
        return data, retained

    @staticmethod
    def _mapped_document_column(document: str, mapping: Any) -> str:
        if isinstance(mapping, dict) and document in mapping:
            mapped_value = mapping[document]
        elif callable(mapping):
            mapped_value = mapping(document)
        else:
            return document
        return mapped_value if isinstance(mapped_value, str) else document

    # Commonly accessed convenience properties (explicit to avoid delegation surprises)
    @property
    def shape(self) -> tuple[int, int]:
        height = self._lazyframe_height(self.data)
        return (height, self.data.collect_schema().len())

    @property
    def data(self) -> pl.LazyFrame:
        return self._data

    @data.setter
    def data(self, value: pl.LazyFrame) -> None:
        if not isinstance(value, pl.LazyFrame):
            raise TypeError(
                "Node data must be a polars LazyFrame "
                f"(received {type(value).__name__})."
            )

        if hasattr(self, "_data"):
            current = self._data
            if current is value:
                return

            self._undo_stack.append(current)
            if len(self._undo_stack) > self.MAX_UNDO_DEPTH:
                self._undo_stack.pop(0)
            self._redo_stack.clear()

        self._data = value

    @property
    def columns(self):  # pragma: no cover
        return self.data.collect_schema().names()

    @property
    def children(self) -> list["Node"]:
        if self.workspace is None:
            return []
        return [
            candidate
            for candidate in self.workspace.nodes.values()
            if any(self._parent_matches(parent, self) for parent in candidate.parents)
        ]

    # ------------------------------------------------------------------
    # Explicit graph-producing dataframe operations
    # ------------------------------------------------------------------
    def filter(self, *predicates: Any, **constraints: Any) -> "Node":
        result = self.data.filter(*predicates, **constraints)
        return self._child_node(
            data=result,
            name=f"filter_{self.name}",
            operation="filter",
        )

    def select(self, *exprs: Any, **named_exprs: Any) -> "Node":
        result = self.data.select(*exprs, **named_exprs)
        # User-driven select may drop source columns; keep derived metadata
        # only when the derived column itself or its source column survives.
        result_columns = set(result.collect_schema().names())
        return self._child_node(
            data=result,
            name=f"select_{self.name}",
            operation="select",
            derived=self._derived_for_columns(self.derived, result_columns),
        )

    def join(
        self,
        other: "Node",
        on: Any = None,
        how: Literal[
            "inner",
            "left",
            "right",
            "full",
            "semi",
            "anti",
            "cross",
            "outer",
        ] = "inner",
        **kwargs: Any,
    ) -> "Node":
        result = self.data.join(other.data, on=on, how=how, **kwargs)
        # Union derived metadata from both sides; result column set may drop
        # entries if join columns collide. Conflicting metadata for the same
        # retained derived column is ambiguous, so reject it instead of letting
        # the right side overwrite the left side silently.
        result_columns = set(result.collect_schema().names())
        merged: dict[str, DerivedColumnMeta] = {}
        for source in (self.derived, other.derived):
            for name, meta in source.items():
                if name not in result_columns:
                    continue
                existing = merged.get(name)
                if existing is not None and existing != meta:
                    raise ValueError(
                        f"Conflicting derived metadata for joined column {name!r}."
                    )
                merged[name] = meta
        return self._child_node(
            data=result,
            name=f"join_{self.name}_{other.name}",
            parents=[self, other],
            operation=f"join({how})",
            derived=merged,
        )

    def slice(self, offset: int, length: int | None = None) -> "Node":
        result = self.data.slice(offset, length)
        return self._child_node(
            data=result,
            name=f"slice_{self.name}",
            operation="slice",
        )

    def drop(
        self,
        columns: Any,
        *more_columns: Any,
        strict: bool = True,
    ) -> "Node":
        """Drop columns using Polars semantics and return a child node.

        Mirrors ``polars.LazyFrame.drop`` while preserving DocWorkspace lineage.
        Cascade rule (decision 7): when a user column is dropped, any derived
        columns whose ``source_column`` matched are also dropped and removed
        from ``Node.derived``.
        """
        before_names = set(self.data.collect_schema().names())
        result = self.data.drop(columns, *more_columns, strict=strict)
        after_names = set(result.collect_schema().names())
        result, retained_derived = self._drop_derived_from_stale_sources(
            result,
            self.derived,
            before_names - after_names,
        )

        document = None
        if self.document and not (
            self.document in before_names and self.document not in after_names
        ):
            document = self.document

        return self._child_node(
            data=result,
            name=f"drop_{self.name}",
            operation="drop",
            derived=retained_derived,
            document=document,
        )

    def rename(self, mapping: Any, *, strict: bool = True) -> "Node":
        """Rename columns in-place using Polars semantics and return this node.

        Cascade rule (decision 7): renaming a source column makes any derived
        columns referencing it stale — they are dropped from the LazyFrame and
        from ``Node.derived``. Users can re-tokenise after the rename.
        """
        before_names = set(self.data.collect_schema().names())
        new_data = self.data.rename(mapping, strict=strict)
        after_names = set(new_data.collect_schema().names())
        new_data, self.derived = self._drop_derived_from_stale_sources(
            new_data,
            self.derived,
            before_names - after_names,
        )

        self.data = new_data

        if self.document:
            self.document = self._mapped_document_column(self.document, mapping)

        return self

    def undo(self) -> "Node":
        if not self._undo_stack:
            raise ValueError("Nothing to undo")

        self._redo_stack.append(self._data)
        self._data = self._undo_stack.pop()
        return self

    def redo(self) -> "Node":
        if not self._redo_stack:
            raise ValueError("Nothing to redo")

        self._undo_stack.append(self._data)
        if len(self._undo_stack) > self.MAX_UNDO_DEPTH:
            self._undo_stack.pop(0)
        self._data = self._redo_stack.pop()
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def document(self) -> str | None:
        return self._document_column

    @document.setter
    def document(self, value: str | None) -> None:
        self._document_column = value

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    # ------------------------------------------------------------------
    # Derived-column metadata (Phase 2, decision 7)
    # ------------------------------------------------------------------
    def register_derived_column(
        self, column_name: str, meta: DerivedColumnMeta
    ) -> None:
        """Record metadata for a hidden derived column on this node.

        The derived column may be physical, or it may be a cache-backed spec
        that analyses hydrate temporarily. This method only writes the metadata
        index.
        """
        self.derived[column_name] = cast(DerivedColumnMeta, dict(meta))

    def unregister_derived_column(self, column_name: str) -> bool:
        """Remove the metadata entry for ``column_name``. Does not touch the
        LazyFrame schema. Returns True if an entry was removed.
        """
        return self.derived.pop(column_name, None) is not None

    def find_derived_column(
        self,
        source_column: str,
        *,
        form: str = "tokens",
        model: str | None = None,
    ) -> str | None:
        """Return the name of a derived column for ``source_column``, or None.

        Filters by ``form`` (default ``"tokens"``); if ``model`` is given,
        further narrows to that backend. When multiple candidates match,
        returns the first by insertion order.
        """
        for name, meta in self.derived.items():
            if meta.get("source_column") != source_column:
                continue
            if meta.get("form") != form:
                continue
            if model is not None and meta.get("model") != model:
                continue
            return name
        return None

    # ------------------------------------------------------------------
    # Schema utilities
    # ------------------------------------------------------------------
    def json_schema(self) -> dict[str, str]:
        """Return raw schema - JSON conversion should be handled by API layer."""
        schema = self.data.collect_schema()
        return {col: str(dtype) for col, dtype in schema.items()} if schema else {}

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        """Get JSON-safe node information suitable for API responses.

        All values are plain Python types (str, int, list, dict, None)
        so the result can be returned directly by FastAPI without
        additional conversion.
        """
        schema = self.data.collect_schema()
        height = self._lazyframe_height(self.data)
        return {
            "id": self.id,
            "name": self.name,
            "operation": self.operation,
            "parent_ids": [self._parent_id(parent) for parent in self.parents],
            "child_ids": [c.id for c in self.children],
            "document": self.document,
            "shape": (height, self.data.collect_schema().len()),
            "schema": {col: str(dtype) for col, dtype in schema.items()},
            "columns": list(schema.names()),
            "can_undo": self.can_undo,
            "can_redo": self.can_redo,
        }

    def to_dict(self, *, base_dir: str | Path | None = None) -> dict[str, Any]:
        from .io import to_dict

        return to_dict(self, base_dir=base_dir)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        workspace: Workspace | None = None,
        base_dir: str | Path | None = None,
    ) -> "Node":
        from .io import from_dict

        return from_dict(payload, workspace=workspace, base_dir=base_dir)

    # Representation --------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Node(id={self.id[:8]}, name='{self.name}', dtype={type(self.data).__name__}, "
            f"parents={len(self.parents)}, children={len(self.children)}, document={self.document})"
        )


__all__ = ["Node", "DerivedColumnMeta"]
