"""Workspace save-time garbage collection and load-time path rebasing."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import polars as pl
from docworkspace.node import Node, TokenizationMeta
from docworkspace.workspace import Workspace
from docworkspace.workspace.io import rebase_workspace_sources
from polars_source_utils import list_source_paths


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


def test_save_keeps_dotfile_parquets(tmp_path: Path):
    """Dotfile parquets are out-of-band caches and must survive GC.

    Analysis side-effect parquets (e.g.
    ``.materialized_concordance_<task_id>_<node_id>.parquet``) are owned by
    background analysis tasks, not by workspace nodes. Their lifecycle is
    managed by the analysis_cache module; the workspace GC must leave them
    alone.
    """
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    data_dir = ws_root / "data"
    data_dir.mkdir()

    kept_parquet = data_dir / "kept.parquet"
    _make_parquet(kept_parquet, pl.DataFrame({"a": [1]}))

    cache_parquet = data_dir / ".materialized_concordance_TASK_NODE.parquet"
    _make_parquet(cache_parquet, pl.DataFrame({"b": [2]}))

    ws = Workspace(name="dotfile_ws", ws_root_dir=ws_root)
    ws.add_node(Node(data=pl.scan_parquet(kept_parquet.resolve()), name="kept"))
    ws.save(ws_root)

    assert cache_parquet.exists(), (
        "Dotfile parquet must survive workspace.save() GC — it's a cache "
        "owned by an analysis task, not the workspace."
    )


def test_rebase_then_load_after_workspace_move(tmp_path: Path):
    """Simulate the real backend flow: rebase plbin files, then load workspace."""

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

    # Rebase on disk BEFORE loading (mirrors backend set_current_workspace).
    rebase_workspace_sources(moved_root)

    ws_reloaded = Workspace.load(moved_root)
    assert len(ws_reloaded.nodes) == 1
    node = next(iter(ws_reloaded.nodes.values()))

    collected = cast(pl.DataFrame, node.data.collect())
    assert collected.to_series().to_list() == [10, 20, 30]

    plbin_files = list((moved_root / "data").glob("*.plbin"))
    sources = list_source_paths(plbin_files[0])
    moved_parquet = (moved_root / "data" / "input.parquet").resolve()
    assert all(Path(s) == moved_parquet for s in sources)


def test_rebase_then_rename_then_save_keeps_parquet(tmp_path: Path):
    """Repro for the reported bug: rename folder after save, then load + unload.

    Sequence:
      1. Save workspace in folder A.
      2. Rename folder A → B (simulating user rename).
      3. rebase_workspace_sources(B)  →  plbin paths point at B/data/…
      4. Load workspace from B.
      5. Save workspace back to B.
      6. Parquet referenced by the node must survive GC.
    """

    folder_a = tmp_path / "Test"
    folder_a.mkdir()
    data_dir = folder_a / "data"
    data_dir.mkdir()

    parquet_path = data_dir / "my_data.parquet"
    _make_parquet(parquet_path, pl.DataFrame({"v": [100, 200]}))

    ws = Workspace(name="Test", ws_root_dir=folder_a)
    ws.add_node(Node(data=pl.scan_parquet(parquet_path.resolve()), name="node"))
    ws.save(folder_a)

    # User renames the folder on disk.
    folder_b = tmp_path / "Test_New"
    folder_a.rename(folder_b)

    # Backend flow: rebase → load → (later) save.
    rebase_workspace_sources(folder_b)
    ws2 = Workspace.load(folder_b)
    assert len(ws2.nodes) == 1

    # Verify data is accessible after rename.
    node = next(iter(ws2.nodes.values()))
    assert cast(pl.DataFrame, node.data.collect()).to_series().to_list() == [100, 200]

    # Save (triggers GC) — the parquet must survive.
    ws2.save(folder_b)
    assert (folder_b / "data" / "my_data.parquet").exists()


def test_rebase_preserves_tokenized_node_after_move(tmp_path: Path):
    """Phase 2.9 regression: a node with a List[Struct] tokens column must
    survive workspace-folder move + rebase_workspace_sources. The rebasing
    walks scan-source paths inside the plbin, not the dataframe schema, so
    it should be schema-agnostic — this test locks that in."""

    folder_a = tmp_path / "Tokens"
    folder_a.mkdir()
    data_dir = folder_a / "data"
    data_dir.mkdir()

    parquet_path = data_dir / "docs.parquet"
    _make_parquet(parquet_path, pl.DataFrame({"text": ["doc one", "doc two"]}))

    ws = Workspace(name="Tokens", ws_root_dir=folder_a)
    base_node = Node(
        data=pl.scan_parquet(parquet_path.resolve()),
        name="docs",
    )
    ws.add_node(base_node)

    # Synthesize a hydrated tokens column on top via with_columns. The metadata
    # is cache-like and tokenization-specific, but the LazyFrame plan remains
    # schema-agnostic for source path rebasing.
    tokens_name = "tokenization.text.lindera:jieba"
    tokenization_meta: TokenizationMeta = {
        "column_name": tokens_name,
        "model": "lindera:jieba",
        "language": "zh",
        "params": {"lowercase": True, "remove_punct": True},
    }
    tokens_frame = base_node.data.with_columns(
        pl.lit(
            [
                {"token": "doc", "start": 0, "end": 3},
                {"token": "one", "start": 4, "end": 7},
            ]
        ).alias(tokens_name)
    )
    tokens_node = Node(
        data=tokens_frame,
        name="docs_tokens",
        parents=[base_node],
        operation="tokenize",
        tokenization={"text": tokenization_meta},
    )
    ws.add_node(tokens_node)
    ws.save(folder_a)

    # Move the workspace folder to a new location.
    folder_b = tmp_path / "Tokens_Moved"
    shutil.copytree(folder_a, folder_b)
    shutil.rmtree(folder_a)

    rebase_workspace_sources(folder_b)
    ws2 = Workspace.load(folder_b)

    # Both nodes should be back, and the tokens node's lineage + metadata
    # preserved.
    assert len(ws2.nodes) == 2
    loaded_tokens_node = next(n for n in ws2.nodes.values() if n.name == "docs_tokens")
    assert loaded_tokens_node.tokenization == {"text": tokenization_meta}
    assert loaded_tokens_node.operation == "tokenize"

    # The List[Struct] column should still be loadable end-to-end.
    collected = cast(pl.DataFrame, loaded_tokens_node.data.collect())
    assert tokens_name in collected.columns
    assert collected.height == 2
