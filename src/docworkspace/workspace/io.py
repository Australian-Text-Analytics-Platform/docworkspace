"""Workspace file persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from polars_source_utils import list_source_paths, replace_source_paths

if TYPE_CHECKING:  # pragma: no cover
    from .core import Workspace

from ..node.io import NODE_DATA_DIR
from ..node.io import from_dict as node_from_dict
from ..node.io import to_dict as node_to_dict


def _resolve_metadata_path(path: Path) -> Path:
    if path.is_dir():
        return path / "metadata.json"
    if path.suffix.lower() == ".json":
        return path
    raise ValueError("Workspace path must be a directory or a .json file")


def _collect_referenced_sources(plbin_paths: Iterable[Path]) -> set[Path]:
    """Return the set of absolute source paths referenced by every plbin plan."""

    referenced: set[Path] = set()
    for plbin in plbin_paths:
        for raw in list_source_paths(plbin):
            referenced.add(Path(raw).resolve())
    return referenced


def _rebase_plan_sources(plbin_path: Path, data_dir: Path) -> None:
    """Rewrite scan source paths inside ``plbin_path`` to ``data_dir`` by basename.

    Only paths whose basename exists under ``data_dir`` and whose stored value
    differs from the resolved target are rewritten. Paths that already match
    the current workspace layout are left untouched.
    """

    current_sources = list_source_paths(plbin_path)
    if not current_sources:
        return

    mapping: dict[str, str] = {}
    for old in current_sources:
        target = (data_dir / Path(old).name).resolve()
        if not target.exists():
            continue
        target_str = str(target)
        if old == target_str:
            continue
        mapping[old] = target_str

    if mapping:
        replace_source_paths(plbin_path, mapping)


def _garbage_collect_workspace_data(
    ws_root_dir: Path, nodes_data: list[dict[str, Any]]
) -> None:
    """Remove unreferenced parquet and plbin files from the workspace data dir.

    * Parquet files not referenced by any registered node's plan are deleted.
    * Plbin files whose name does not match a registered node's ``data_path``
      are deleted (they belong to nodes no longer in the workspace).

    Dotfiles are **always skipped**. By convention they are out-of-band caches
    (e.g. analysis side-effect parquets under
    ``.materialized_<feature>_<task_id>_<node_id>.parquet``) whose lifecycle
    is managed by their creators, not by the workspace.
    """

    data_dir = ws_root_dir / NODE_DATA_DIR
    if not data_dir.exists() or not data_dir.is_dir():
        return

    expected_plbin_names = {
        Path(str(node_payload["data_path"])).name for node_payload in nodes_data
    }
    registered_plbins = [
        data_dir / name for name in expected_plbin_names if (data_dir / name).exists()
    ]

    referenced_sources = _collect_referenced_sources(registered_plbins)

    for candidate in data_dir.iterdir():
        if not candidate.is_file():
            continue
        if candidate.name.startswith("."):
            continue
        suffix = candidate.suffix.lower()
        if suffix == ".plbin":
            if candidate.name not in expected_plbin_names:
                candidate.unlink(missing_ok=True)
        elif suffix == ".parquet":
            if candidate.resolve() not in referenced_sources:
                candidate.unlink(missing_ok=True)


def write_workspace(workspace: "Workspace", path: str | Path) -> None:
    target = _resolve_metadata_path(Path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace.ws_root_dir = target.parent

    description = workspace.description
    created_at = workspace.created_at
    modified_at = workspace.modified_at

    nodes_data: list[dict[str, Any]] = []
    for node in workspace.nodes.values():
        nodes_data.append(node_to_dict(node))

    workspace_metadata: dict[str, Any] = {
        "id": workspace.id,
        "name": workspace.name,
        "version": 2,
        "description": description,
        "created_at": created_at,
        "modified_at": modified_at,
    }

    data = {"workspace_metadata": workspace_metadata, "nodes": nodes_data}
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    _garbage_collect_workspace_data(target.parent, nodes_data)


def read_workspace_metadata(path: str | Path) -> dict[str, Any]:
    """Load and return the workspace metadata dictionary from metadata.json.

    This helper only reads/parses the JSON metadata file and does not attempt
    to load any node data payload files.
    """

    target = _resolve_metadata_path(Path(path))
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_workspace(path: str | Path) -> "Workspace":
    from .core import Workspace

    target = _resolve_metadata_path(Path(path))
    data = read_workspace_metadata(path)

    ws_meta = data.get("workspace_metadata", {})
    workspace = Workspace(
        name=ws_meta.get("name", "restored_workspace"),
        ws_root_dir=target.parent,
    )
    workspace.id = ws_meta.get("id", workspace.id)
    workspace.description = ws_meta.get("description", "") or ""
    workspace.created_at = ws_meta.get("created_at")
    workspace.modified_at = ws_meta.get("modified_at")

    for node_entry in data.get("nodes", []):
        workspace.add_node(node_from_dict(node_entry, workspace=workspace))

    return workspace


def rebase_workspace_sources(path: str | Path) -> None:
    """Rewrite stale scan source paths inside every plbin listed in ``metadata.json``.

    Call this **after** the workspace folder has reached its final location on
    disk (i.e. after any rename / move) but **before** deserializing the
    workspace nodes into memory, so that ``LazyFrame.deserialize`` sees paths
    that match the current filesystem.
    """

    target = _resolve_metadata_path(Path(path))
    data = read_workspace_metadata(path)
    data_dir = (target.parent / NODE_DATA_DIR).resolve()

    for node_entry in data.get("nodes", []):
        plbin_abs = (target.parent / Path(str(node_entry["data_path"]))).resolve()
        if plbin_abs.exists():
            _rebase_plan_sources(plbin_abs, data_dir)


__all__ = [
    "write_workspace",
    "read_workspace_metadata",
    "read_workspace",
    "rebase_workspace_sources",
]
