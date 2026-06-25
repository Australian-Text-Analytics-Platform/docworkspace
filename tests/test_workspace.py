"""Tests for the Workspace class."""

import tempfile
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from docworkspace import Node, Workspace


class TestWorkspace:
    """Test cases for the Workspace class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample polars DataFrame."""
        return pl.DataFrame({"text": ["Hello", "World", "Test"], "value": [1, 2, 3]})

    @pytest.fixture
    def workspace(self):
        """Create a test workspace."""
        return Workspace("test_workspace")

    def test_workspace_creation(self):
        """Test creating a Workspace."""
        workspace = Workspace("test_workspace")
        assert workspace.name == "test_workspace"
        assert len(workspace.nodes) == 0
        assert workspace.id is not None
        assert workspace.ws_root_dir.exists()

    def test_add_node(self, workspace, sample_df):
        """Test adding a node to workspace."""
        node = Node(sample_df.lazy(), "test_node", workspace)

        # Node should already be in workspace due to constructor
        assert len(workspace.nodes) == 1
        assert node.id in workspace.nodes
        assert workspace.nodes[node.id] == node

    def test_get_node_by_name(self, workspace, sample_df):
        """Test getting a node by name."""
        node = workspace.add_node(
            Node(data=sample_df.lazy(), name="test_data", workspace=workspace)
        )

        found_node = workspace.get_node_by_name("test_data")
        assert found_node == node

        not_found = workspace.get_node_by_name("nonexistent")
        assert not_found is None

    def test_get_root_nodes(self, workspace, sample_df):
        """Test getting root nodes (nodes without parents)."""
        root_node = workspace.add_node(
            Node(data=sample_df.lazy(), name="root", workspace=workspace)
        )
        child_node = root_node.filter(pl.col("value") > 1)

        root_nodes = workspace.get_root_nodes()

        assert len(root_nodes) == 1
        assert root_nodes[0] == root_node
        assert child_node not in root_nodes

    def test_get_leaf_nodes(self, workspace, sample_df):
        """Test getting leaf nodes (nodes without children)."""
        root_node = workspace.add_node(
            Node(data=sample_df.lazy(), name="root", workspace=workspace)
        )
        child_node = root_node.filter(pl.col("value") > 1)

        leaf_nodes = workspace.get_leaf_nodes()

        assert len(leaf_nodes) == 1
        assert leaf_nodes[0] == child_node
        assert root_node not in leaf_nodes

    def test_workspace_info_json(self, workspace, sample_df):
        """Test workspace info_json payload."""
        # Create some nodes
        root1 = workspace.add_node(
            Node(data=sample_df.lazy(), name="root1", workspace=workspace)
        )
        root2 = workspace.add_node(
            Node(data=sample_df.lazy(), name="root2", workspace=workspace)
        )
        root1.filter(pl.col("value") > 1)
        root2.filter(pl.col("value") > 2)

        summary = workspace.info_json()

        assert summary["total_nodes"] == 4
        assert summary["root_nodes"] == 2
        assert summary["leaf_nodes"] == 2
        assert "description" in summary
        assert "created_at" in summary
        assert "modified_at" in summary


class TestWorkspaceSerialization:
    """Test workspace serialization and deserialization."""

    @pytest.fixture
    def populated_workspace(self):
        """Create a workspace with some nodes and relationships."""
        workspace = Workspace("test_workspace")
        workspace.description = "serialized workspace"
        workspace.created_at = "2024-01-01T00:00:00Z"
        workspace.modified_at = "2024-01-01T12:00:00Z"

        # Create nodes
        df1 = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "category": ["A", "B", "A"],
                "value": [10, 20, 30],
            }
        )

        df2 = pl.DataFrame({"id": [1, 2, 3], "extra": ["x", "y", "z"]})

        root1 = workspace.add_node(
            Node(data=df1.lazy(), name="root1", workspace=workspace)
        )
        root2 = workspace.add_node(
            Node(data=df2.lazy(), name="root2", workspace=workspace)
        )

        # Create relationships
        root1.filter(pl.col("category") == "A")
        root1.join(root2, on="id")

        return workspace

    def test_workspace_serialization_roundtrip(self, populated_workspace):
        """Round-trip workspace serialization using JSON format (pickle removed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "metadata.json"

            # Serialize
            populated_workspace.save(meta_path)

            # Deserialize
            loaded_workspace = Workspace.load(meta_path)

            # Check workspace properties
            assert loaded_workspace.name == populated_workspace.name
            assert len(loaded_workspace.nodes) == len(populated_workspace.nodes)
            assert loaded_workspace.description == "serialized workspace"
            assert loaded_workspace.created_at == "2024-01-01T00:00:00Z"
            assert loaded_workspace.modified_at == "2024-01-01T12:00:00Z"

            # Check nodes exist
            root1 = loaded_workspace.get_node_by_name("root1")
            root2 = loaded_workspace.get_node_by_name("root2")
            assert root1 is not None
            assert root2 is not None

            # Check relationships are preserved
            assert len(root1.children) == 2  # filtered and merged
            assert len(root2.children) == 1  # merged

    def test_serialization_with_lazy_nodes(self):
        """Test serialization of workspace containing lazy nodes."""
        workspace = Workspace("lazy_workspace")

        # Create lazy nodes
        lazy_df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        lazy_node = workspace.add_node(
            Node(data=lazy_df, name="lazy_node", workspace=workspace)
        )
        lazy_node.filter(pl.col("a") > 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "metadata.json"

            # Serialize (JSON format only)
            workspace.save(meta_path)

            # Deserialize
            loaded_workspace = Workspace.load(meta_path)

            # Check nodes
            loaded_lazy = loaded_workspace.get_node_by_name("lazy_node")
            assert loaded_lazy is not None
            # After serialization, lazy frames should remain lazy
            assert isinstance(loaded_lazy.data, pl.LazyFrame)
            assert cast(pl.DataFrame, loaded_lazy.data.collect()).to_dict(
                as_series=False
            ) == {"a": [1, 2, 3], "b": [4, 5, 6]}

    def test_undo_redo_stacks_are_not_persisted(self):
        """Undo/redo history is in-memory only and must reset after load."""
        workspace = Workspace("undo_runtime_only")
        node = workspace.add_node(
            Node(
                data=pl.DataFrame({"a": [1, 2, 3]}).lazy(),
                name="root",
                workspace=workspace,
            )
        )

        node.data = node.data.with_columns(pl.lit(1).alias("b"))
        assert node.can_undo is True

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "metadata.json"
            workspace.save(meta_path)

            loaded_workspace = Workspace.load(meta_path)
            loaded_node = loaded_workspace.get_node_by_name("root")

            assert loaded_node is not None
            assert loaded_node.can_undo is False
            assert loaded_node.can_redo is False

    def test_workspace_serialized_file_structure(self, populated_workspace):
        """Validate on-disk JSON structure contains expected envelope keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "metadata.json"
            populated_workspace.save(meta_path)
            import json as _json

            with open(meta_path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            assert "workspace_metadata" in data
            assert "nodes" in data
            assert isinstance(data["nodes"], list)
            # Ensure each node entry has required composite sections
            for n in data["nodes"]:
                assert "node_metadata" in n
                assert "data_path" in n
                assert "serialized_data" not in n
                rel_path = n["data_path"]
                assert isinstance(rel_path, str)
                abs_path = (Path(tmpdir) / rel_path).resolve()
                assert abs_path.exists(), f"Missing node data file: {abs_path}"
                assert abs_path.stat().st_size > 0

    def test_remove_node_deletes_binary_file_when_workspace_dir_attached(self):
        """Removing a node should delete its persisted data/<node_id>.plbin file."""
        workspace = Workspace("ws")
        df = pl.DataFrame({"a": [1, 2, 3]})
        node = workspace.add_node(Node(data=df.lazy(), name="n", workspace=workspace))

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "metadata.json"
            workspace.save(meta_path)
            workspace.ws_root_dir = Path(tmpdir)

            payload_file = Path(tmpdir) / "data" / f"{node.id}.plbin"
            assert payload_file.exists()

            assert workspace.remove_node(node.id) is True
            assert not payload_file.exists()


class TestWorkspaceGraphOperations:
    """Test workspace graph analysis and relationship operations."""

    @pytest.fixture
    def complex_workspace(self):
        """Create a workspace with multiple nodes and relationships."""
        workspace = Workspace("complex")

        # Create initial data
        df1 = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [2, 3, 4], "score": [0.5, 0.7, 0.9]})

        root1 = workspace.add_node(Node(df1.lazy(), "root1"))
        root2 = workspace.add_node(Node(df2.lazy(), "root2"))

        # Create derived nodes
        filtered1 = root1.filter(pl.col("value") > 15)
        filtered2 = root2.filter(pl.col("score") > 0.6)

        # Create a joined node (has multiple parents)
        _joined = filtered1.join(filtered2, on="id", how="inner")

        return workspace

    def test_workspace_graph_structure(self, complex_workspace):
        """Test the generic graph structure generation."""
        graph_data = complex_workspace.graph_json()

        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "workspace_info" not in graph_data

        # Check node data structure
        if graph_data["nodes"]:
            node_data = graph_data["nodes"][0]
            required_fields = [
                "id",
                "name",
                "operation",
            ]
            for field in required_fields:
                assert field in node_data

    def test_workspace_graph_survives_broken_node_info(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """One node failing `info()` must not break the whole graph payload."""
        workspace = Workspace("graph_resilience")
        good_node = Node(
            data=pl.DataFrame({"x": [1, 2, 3]}).lazy(),
            name="good",
            workspace=workspace,
        )
        bad_node = Node(
            data=pl.DataFrame({"y": [4, 5]}).lazy(),
            name="bad",
            workspace=workspace,
        )

        # Simulate a broken lazy plan / missing source file: info() raises.
        def _boom() -> dict[str, Any]:
            raise RuntimeError("source parquet missing")

        monkeypatch.setattr(bad_node, "info", _boom)

        graph_data = workspace.graph_json()

        nodes_by_id = {n["id"]: n for n in graph_data["nodes"]}
        assert good_node.id in nodes_by_id
        assert bad_node.id in nodes_by_id
        # Healthy node still carries its real info.
        assert "shape" in nodes_by_id[good_node.id]
        # Broken node carries an error envelope plus identity fields.
        assert nodes_by_id[bad_node.id]["name"] == "bad"
        assert "error" in nodes_by_id[bad_node.id]
        assert "RuntimeError" in nodes_by_id[bad_node.id]["error"]

    def test_node_workspace_transfer(self):
        """Test moving nodes between workspaces."""
        workspace1 = Workspace("ws1")
        workspace2 = Workspace("ws2")

        df = pl.DataFrame({"col": [1, 2, 3]})
        node = Node(df.lazy(), "test_node", workspace1)

        # Node should be in workspace1
        assert node.id in workspace1.nodes
        assert node.workspace == workspace1

        # Add to workspace2 (should move from workspace1)
        workspace2.add_node(node)

        assert node.id not in workspace1.nodes
        assert node.id in workspace2.nodes
        assert node.workspace == workspace2

    def test_workspace_boolean_and_len_operations(self):
        """Test workspace boolean evaluation and length operations."""
        workspace = Workspace("bool_test")

        # Empty workspace should still be truthy
        assert bool(workspace) is True
        assert len(workspace) == 0

        # Add a node
        df = pl.DataFrame({"col": [1]})
        workspace.add_node(Node(df.lazy(), "test"))

        assert bool(workspace) is True
        assert len(workspace) == 1

    def test_remove_node_rewires_child_to_all_parents(self):
        """Deleting an intermediate node should preserve lineage via parent inheritance."""
        workspace = Workspace("rewire_test")

        df_left = pl.DataFrame({"id": [1, 2], "left": [10, 20]})
        df_right = pl.DataFrame({"id": [1, 2], "right": [30, 40]})

        node_b = workspace.add_node(Node(df_left.lazy(), "B"))
        node_c = workspace.add_node(Node(df_right.lazy(), "C"))

        node_a = node_b.join(node_c, on="id", how="inner")
        node_d = node_a.filter(pl.col("id") > 0)

        assert node_a in node_b.children
        assert node_a in node_c.children
        assert node_b in node_a.parents
        assert node_c in node_a.parents
        assert node_d in node_a.children
        assert node_a in node_d.parents

        assert workspace.remove_node(node_a.id) is True

        assert node_d in node_b.children
        assert node_d in node_c.children
        assert node_b in node_d.parents
        assert node_c in node_d.parents
        assert node_a not in node_d.parents
        assert node_a not in workspace.nodes.values()
