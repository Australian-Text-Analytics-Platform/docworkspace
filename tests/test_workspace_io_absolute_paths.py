"""Workspace save-time garbage collection and load-time path rebasing."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import polars as pl
from polars_text import list_source_paths

from docworkspace.node import Node
from docworkspace.workspace import Workspace


def _make_parquet(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_save_removes_orphan_parquet_and_plbin(tmp_path: Path):
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    data_dir = ws_root / "data"
    data_dir.mkdir()

    kept_parquet = data_dir / "kept.parquet"
    _make_parquet(kept_parquet, pl.DataFrame({"a": [1, 2, 3]}))

    orphan_parquet = data_dir / "orphan.parquet"
    _make_parquet(orphan_parquet, pl.DataFrame({"b": [9]}))

    orphan_plbin = data_dir / "ghost-node-id.plbin"
    orphan_plbin.write_bytes(b"not-a-real-plan")

    ws = Workspace(name="gc_ws", ws_root_dir=ws_root)
    lazy = pl.scan_parquet(kept_parquet.resolve())
    ws.add_node(Node(data=lazy, name="kept"))

    ws.save(ws_root)

    assert kept_parquet.exists(), "Referenced parquet must be kept"
    assert not orphan_parquet.exists(), "Unreferenced parquet must be deleted"
    assert not orphan_plbin.exists(), "Unregistered plbin must be deleted"

    plbin_files = list(data_dir.glob("*.plbin"))
    assert len(plbin_files) == 1
    sources = list_source_paths(plbin_files[0])
    assert any(Path(s) == kept_parquet.resolve() for s in sources)


def test_load_rebases_source_paths_after_workspace_move(tmp_path: Path):
    original_root = tmp_path / "original"
    original_root.mkdir()
    data_dir = original_root / "data"
    data_dir.mkdir()

    parquet_path = data_dir / "input.parquet"
    _make_parquet(parquet_path, pl.DataFrame({"x": [10, 20, 30]}))

    ws = Workspace(name="move_ws", ws_root_dir=original_root)
    lazy = pl.scan_parquet(parquet_path.resolve())
    ws.add_node(Node(data=lazy, name="input"))
    ws.save(original_root)

    moved_root = tmp_path / "moved"
    shutil.copytree(original_root, moved_root)
    shutil.rmtree(original_root)

    ws_reloaded = Workspace.load(moved_root)
    assert len(ws_reloaded.nodes) == 1
    node = next(iter(ws_reloaded.nodes.values()))

    collected = cast(pl.DataFrame, node.data.collect())
    assert collected.to_series().to_list() == [10, 20, 30]

    plbin_files = list((moved_root / "data").glob("*.plbin"))
    sources = list_source_paths(plbin_files[0])
    moved_parquet = (moved_root / "data" / "input.parquet").resolve()
    assert all(Path(s) == moved_parquet for s in sources)
