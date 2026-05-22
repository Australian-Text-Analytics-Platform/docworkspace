import json
import os
from pathlib import Path
from typing import cast

import polars as pl
from docworkspace.node.io import dumps, from_dict, loads, to_dict

from docworkspace import DerivedColumnMeta, Node, Workspace


def test_node_to_dict_persists_lazyframe_payload(tmp_path: Path):
    workspace = Workspace("node_io")
    workspace.ws_root_dir = tmp_path
    node = workspace.add_node(
        Node(
            data=pl.DataFrame({"text": ["a", "b"], "value": [1, 2]}).lazy(),
            name="root",
            workspace=workspace,
            operation="source",
        )
    )
    node.document = "text"

    payload = to_dict(node, base_dir=tmp_path)

    assert payload == {
        "node_metadata": {
            "id": node.id,
            "name": "root",
            "operation": "source",
            "document": "text",
            "derived": {},
            "parents": [],
        },
        "data_path": f"data/{node.id}.plbin",
    }

    data_file = tmp_path / payload["data_path"]
    assert data_file.exists()
    restored = pl.LazyFrame.deserialize(data_file.open("rb"), format="binary")
    assert cast(pl.DataFrame, restored.collect()).to_dict(as_series=False) == {
        "text": ["a", "b"],
        "value": [1, 2],
    }


def test_node_exposes_instance_and_class_serialization_helpers(tmp_path: Path):
    workspace = Workspace("node_io")
    workspace.ws_root_dir = tmp_path
    node = workspace.add_node(
        Node(
            data=pl.DataFrame({"value": [1, 2, 3]}).lazy(),
            name="root",
            workspace=workspace,
        )
    )

    payload = node.to_dict(base_dir=tmp_path)
    restored = Node.from_dict(
        payload,
        workspace=Workspace("restored", ws_root_dir=tmp_path),
        base_dir=tmp_path,
    )

    assert payload["node_metadata"]["id"] == node.id
    assert restored.id == node.id
    assert cast(pl.DataFrame, restored.data.collect()).to_dict(as_series=False) == {
        "value": [1, 2, 3]
    }


def test_node_dumps_returns_json_payload_and_persists_data_file(tmp_path: Path):
    workspace = Workspace("node_io")
    workspace.ws_root_dir = tmp_path
    node = workspace.add_node(
        Node(
            data=pl.DataFrame({"value": [1, 2, 3]}).lazy(),
            name="root",
            workspace=workspace,
        )
    )

    serialized = dumps(node, base_dir=tmp_path)
    payload = json.loads(serialized)

    assert payload["node_metadata"]["id"] == node.id
    assert payload["data_path"] == f"data/{node.id}.plbin"
    assert (tmp_path / payload["data_path"]).exists()


def test_node_from_dict_restores_node_state(tmp_path: Path):
    source_workspace = Workspace("source")
    source_workspace.ws_root_dir = tmp_path
    node = source_workspace.add_node(
        Node(
            data=pl.DataFrame({"text": ["x", "y"], "value": [10, 20]}).lazy(),
            name="restorable",
            workspace=source_workspace,
            operation="filter",
        )
    )
    node.document = "text"
    payload = to_dict(node, base_dir=tmp_path)

    restored_workspace = Workspace("restored", ws_root_dir=tmp_path)
    restored = from_dict(payload, workspace=restored_workspace, base_dir=tmp_path)

    assert restored.id == node.id
    assert restored.name == "restorable"
    assert restored.operation == "filter"
    assert restored.document == "text"
    assert restored.workspace is restored_workspace
    assert restored.parents == []
    assert restored.children == []
    assert restored.can_undo is False
    assert restored.can_redo is False
    assert cast(pl.DataFrame, restored.data.collect()).to_dict(as_series=False) == {
        "text": ["x", "y"],
        "value": [10, 20],
    }


def test_node_loads_round_trip_from_json_string(tmp_path: Path):
    source_workspace = Workspace("source")
    source_workspace.ws_root_dir = tmp_path
    node = source_workspace.add_node(
        Node(
            data=pl.DataFrame({"value": [3, 4]}).lazy(),
            name="round_trip",
            workspace=source_workspace,
        )
    )

    serialized = dumps(node, base_dir=tmp_path)
    restored = loads(
        serialized,
        workspace=Workspace("restored", ws_root_dir=tmp_path),
        base_dir=tmp_path,
    )

    assert restored.id == node.id
    assert restored.name == "round_trip"
    assert cast(pl.DataFrame, restored.data.collect()).to_dict(as_series=False) == {
        "value": [3, 4]
    }


def test_node_from_dict_uses_constructor_defaults_for_runtime_state(tmp_path: Path):
    source_workspace = Workspace("source")
    source_workspace.ws_root_dir = tmp_path
    node = source_workspace.add_node(
        Node(
            data=pl.DataFrame({"value": [1, 2]}).lazy(),
            name="constructor_restore",
            workspace=source_workspace,
        )
    )
    payload = to_dict(node, base_dir=tmp_path)
    restored = from_dict(
        payload,
        workspace=Workspace("restored", ws_root_dir=tmp_path),
        base_dir=tmp_path,
    )

    restored.data = restored.data.with_columns(pl.lit(9).alias("extra"))
    assert restored.can_undo is True


def test_node_from_dict_restores_existing_parent_nodes_by_id(tmp_path: Path):
    workspace = Workspace("source")
    workspace.ws_root_dir = tmp_path
    parent = workspace.add_node(
        Node(
            data=pl.DataFrame({"value": [1]}).lazy(), name="parent", workspace=workspace
        )
    )
    child = parent.filter(pl.col("value") > 0)

    payload = to_dict(child, base_dir=tmp_path)
    restored_workspace = Workspace("restored", ws_root_dir=tmp_path)
    restored_parent = Node(
        data=pl.DataFrame({"value": [1]}).lazy(),
        name="parent",
        workspace=restored_workspace,
        id=parent.id,
    )

    restored_child = from_dict(payload, workspace=restored_workspace, base_dir=tmp_path)

    assert restored_child.parents == [restored_parent]


def test_node_from_dict_without_workspace_preserves_parent_ids(tmp_path: Path):
    workspace = Workspace("source")
    workspace.ws_root_dir = tmp_path
    parent = workspace.add_node(
        Node(
            data=pl.DataFrame({"value": [1]}).lazy(),
            name="parent",
            workspace=workspace,
        )
    )
    child = Node(
        data=pl.DataFrame({"value": [2]}).lazy(),
        name="child",
        workspace=None,
        parents=[parent.id],
    )

    payload = to_dict(child, base_dir=tmp_path)
    restored = from_dict(payload, base_dir=tmp_path)

    assert restored.workspace is None
    assert restored.parents == [parent.id]


def test_node_from_dict_ignores_missing_parent_ids(tmp_path: Path):
    workspace = Workspace("source")
    workspace.ws_root_dir = tmp_path
    node = workspace.add_node(
        Node(
            data=pl.DataFrame({"value": [1]}).lazy(), name="child", workspace=workspace
        )
    )

    payload = to_dict(node, base_dir=tmp_path)
    payload["node_metadata"]["parents"] = ["missing-parent-id"]
    restored = from_dict(
        payload,
        workspace=Workspace("restored", ws_root_dir=tmp_path),
        base_dir=tmp_path,
    )

    assert restored.parents == []


def test_node_derived_metadata_round_trip(tmp_path: Path):
    """Phase 2.4 v2: Node.derived survives to_dict / from_dict."""
    workspace = Workspace("node_io_derived")
    workspace.ws_root_dir = tmp_path
    derived_name = "__derived__.tokens.text.jieba"
    meta: DerivedColumnMeta = {
        "source_column": "text",
        "form": "tokens",
        "model": "jieba",
        "language": "zh",
        "generated_at": "2026-05-12T00:00:00+00:00",
    }
    node = workspace.add_node(
        Node(
            data=pl.DataFrame({"text": ["今天天气很好"]}).lazy(),
            name="zh_root",
            workspace=workspace,
            operation="source",
            derived={derived_name: meta},
        )
    )
    node.document = "text"

    payload = to_dict(node, base_dir=tmp_path)
    assert payload["node_metadata"]["derived"] == {derived_name: meta}

    # Round-trip into a fresh workspace
    workspace2 = Workspace("node_io_derived_loaded")
    workspace2.ws_root_dir = tmp_path
    restored = from_dict(payload, workspace=workspace2)
    assert restored.derived == {derived_name: meta}
    assert restored.find_derived_column("text") == derived_name
    assert restored.find_derived_column("text", model="jieba") == derived_name
    assert restored.find_derived_column("text", model="other-model") is None


def test_node_legacy_payload_without_derived_loads_with_empty_dict(
    tmp_path: Path,
):
    """Backward compat: workspaces persisted before Phase 2 lacking ``derived``
    must still load, defaulting it to an empty dict."""
    workspace = Workspace("legacy_node_io")
    workspace.ws_root_dir = tmp_path
    node = workspace.add_node(
        Node(
            data=pl.DataFrame({"text": ["legacy"]}).lazy(),
            name="legacy_root",
            workspace=workspace,
            operation="source",
        )
    )

    # Build a "legacy" payload — strip the new field the way old files would.
    payload = to_dict(node, base_dir=tmp_path)
    legacy_metadata = dict(payload["node_metadata"])
    legacy_metadata.pop("derived", None)
    legacy_payload = {**payload, "node_metadata": legacy_metadata}

    workspace2 = Workspace("legacy_loaded")
    workspace2.ws_root_dir = tmp_path
    restored = from_dict(legacy_payload, workspace=workspace2)
    assert restored.derived == {}


def test_node_derived_propagates_through_getattr(tmp_path: Path):
    """Phase 2.4 v2: Node.derived propagates to children spawned by delegated
    LazyFrame methods (schema-preserving ops like .head / .sort)."""
    workspace = Workspace("derive_propagate")
    workspace.ws_root_dir = tmp_path
    derived_name = "__derived__.tokens.text.jieba"
    meta: DerivedColumnMeta = {
        "source_column": "text",
        "form": "tokens",
        "model": "jieba",
        "language": "zh",
        "generated_at": "2026-05-12T00:00:00+00:00",
    }
    parent = workspace.add_node(
        Node(
            data=pl.DataFrame({"text": ["a", "b", "c"]}).lazy(),
            name="zh_parent",
            workspace=workspace,
            operation="source",
            derived={derived_name: meta},
        )
    )
    parent.document = "text"
    child = parent.head(2)
    assert child.derived == {derived_name: meta}


def test_node_drop_cascades_derived_columns(tmp_path: Path):
    """Decision 7: dropping a source column auto-drops any derived columns
    that reference it (both schema and metadata)."""
    workspace = Workspace("derived_drop_cascade")
    workspace.ws_root_dir = tmp_path
    parent_lf = pl.DataFrame(
        {
            "text": ["a", "b"],
            "other": [1, 2],
            "__derived__.tokens.text.jieba": [
                [{"token": "a", "start": 0, "end": 1}],
                [{"token": "b", "start": 0, "end": 1}],
            ],
        }
    ).lazy()
    meta: DerivedColumnMeta = {
        "source_column": "text",
        "form": "tokens",
        "model": "jieba",
        "language": "zh",
        "generated_at": "2026-05-12T00:00:00+00:00",
    }
    parent = workspace.add_node(
        Node(
            data=parent_lf,
            name="parent",
            workspace=workspace,
            derived={"__derived__.tokens.text.jieba": meta},
        )
    )

    # Dropping an UNRELATED column does NOT cascade.
    survivor = parent.drop("other")
    assert "__derived__.tokens.text.jieba" in survivor.derived
    assert "__derived__.tokens.text.jieba" in survivor.data.collect_schema().names()

    # Dropping the SOURCE column cascades: the derived column disappears from
    # both the LazyFrame schema and the metadata index.
    cascaded = parent.drop("text")
    after_names = cascaded.data.collect_schema().names()
    assert "__derived__.tokens.text.jieba" not in after_names
    assert "__derived__.tokens.text.jieba" not in cascaded.derived


def test_node_rename_cascades_derived_columns(tmp_path: Path):
    """Decision 7: renaming a source column drops derived columns that
    referenced it (they become stale; user can re-tokenise)."""
    workspace = Workspace("derived_rename_cascade")
    workspace.ws_root_dir = tmp_path
    parent_lf = pl.DataFrame(
        {
            "text": ["a", "b"],
            "__derived__.tokens.text.jieba": [
                [{"token": "a", "start": 0, "end": 1}],
                [{"token": "b", "start": 0, "end": 1}],
            ],
        }
    ).lazy()
    meta: DerivedColumnMeta = {
        "source_column": "text",
        "form": "tokens",
        "model": "jieba",
        "language": "zh",
        "generated_at": "2026-05-12T00:00:00+00:00",
    }
    node = workspace.add_node(
        Node(
            data=parent_lf,
            name="rename_target",
            workspace=workspace,
            derived={"__derived__.tokens.text.jieba": meta},
        )
    )

    node.rename({"text": "body"})
    after_names = node.data.collect_schema().names()
    assert "body" in after_names
    assert "text" not in after_names
    assert "__derived__.tokens.text.jieba" not in after_names
    assert node.derived == {}
