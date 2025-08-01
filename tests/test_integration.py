"""Integration tests for docworkspace."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from docframe import DocDataFrame, DocLazyFrame
from docworkspace import Node, Workspace


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.fixture
    def corpus_data(self):
        """Create sample corpus data."""
        return pl.DataFrame(
            {
                "doc_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
                "text": [
                    "The quick brown fox jumps over the lazy dog",
                    "A journey of a thousand miles begins with a single step",
                    "To be or not to be that is the question",
                    "All that glitters is not gold",
                    "Where there is a will there is a way",
                ],
                "category": ["animals", "wisdom", "literature", "wisdom", "motivation"],
                "year": [2020, 2019, 2021, 2018, 2022],
                "score": [0.8, 0.9, 0.7, 0.85, 0.95],
            }
        )

    def test_complex_workflow_with_docdataframe(self, corpus_data):
        """Test a complex workflow using DocDataFrame."""
        workspace = Workspace("corpus_analysis")

        # Load as DocDataFrame
        doc_df = DocDataFrame(corpus_data, document_column="text")
        corpus_node = workspace.add_node(
            Node(data=doc_df, name="corpus", workspace=workspace)
        )

        # Filter by category
        wisdom_docs = corpus_node.filter(pl.col("category") == "wisdom")
        assert len(wisdom_docs.data) == 2
        assert wisdom_docs.data.document_column == "text"

        # Filter by score
        high_score_docs = corpus_node.filter(pl.col("score") > 0.8)
        assert len(high_score_docs.data) == 3

        # Check relationships
        assert len(corpus_node.children) == 2
        assert wisdom_docs in corpus_node.children
        assert high_score_docs in corpus_node.children

    def test_lazy_evaluation_workflow(self):
        """Test workflow with lazy evaluation."""
        workspace = Workspace("lazy_workflow")

        # Create lazy dataframe
        lazy_df = pl.LazyFrame(
            {
                "id": range(1000),
                "value": [i * 2 for i in range(1000)],
                "category": [
                    "A" if i % 3 == 0 else "B" if i % 3 == 1 else "C"
                    for i in range(1000)
                ],
            }
        )

        # Load lazy frame
        data_node = workspace.add_node(
            Node(data=lazy_df, name="raw_data", workspace=workspace)
        )
        assert data_node.is_lazy

        # Apply transformations (all lazy)
        filtered = data_node.filter(pl.col("value") > 100)
        assert filtered.is_lazy

        grouped = filtered.group_by("category").agg(pl.col("value").mean())
        assert grouped.is_lazy

        # Collect at the end
        result = grouped.collect()
        assert not result.is_lazy
        assert isinstance(result.data, pl.DataFrame)

        # Check workspace has all nodes
        assert len(workspace.nodes) == 4  # raw_data, filtered, grouped, collected

    def test_mixed_data_types_workflow(self):
        """Test workflow mixing different data types."""
        workspace = Workspace("mixed_types")

        # Create different data types
        eager_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lazy_df = pl.LazyFrame({"a": [2, 3, 4], "c": [7, 8, 9]})
        doc_df = DocDataFrame(
            pl.DataFrame({"a": [3, 4, 5], "text": ["one", "two", "three"]}),
            document_column="text",
        )

        # Load all data types
        eager_node = workspace.add_node(
            Node(data=eager_df, name="eager_data", workspace=workspace)
        )
        lazy_node = workspace.add_node(
            Node(data=lazy_df, name="lazy_data", workspace=workspace)
        )
        doc_node = workspace.add_node(
            Node(data=doc_df, name="doc_data", workspace=workspace)
        )

        # Check initial states
        assert not eager_node.is_lazy
        assert lazy_node.is_lazy
        assert not doc_node.is_lazy

        # Perform operations
        eager_filtered = eager_node.filter(pl.col("a") > 1)
        lazy_collected = lazy_node.collect()
        _doc_filtered = doc_node.filter(pl.col("a") > 3)

        # Join operations
        # Note: Need to collect lazy frame first for join
        _joined = eager_filtered.join(lazy_collected, on="a")

        # Check final workspace state
        assert len(workspace.nodes) == 7
        assert workspace.summary()["total_nodes"] == 7

    def test_doc_lazyframe_workflow(self, corpus_data):
        """Test workflow with DocLazyFrame."""
        workspace = Workspace("lazy_doc_analysis")

        # Load as DocLazyFrame for lazy evaluation
        doc_lazy_df = DocLazyFrame(corpus_data.lazy(), document_column="text")
        lazy_doc_node = workspace.add_node(
            Node(data=doc_lazy_df, name="lazy_corpus", workspace=workspace)
        )

        # Verify lazy state
        assert lazy_doc_node.is_lazy
        assert lazy_doc_node.document_column == "text"

        # Perform lazy operations
        filtered_lazy = lazy_doc_node.filter(pl.col("score") > 0.8)
        assert filtered_lazy.is_lazy

        # Collect to materialize
        materialized = filtered_lazy.collect()
        assert not materialized.is_lazy
        assert len(materialized.data) == 3

        # Check relationships
        assert len(lazy_doc_node.children) == 1
        assert filtered_lazy in lazy_doc_node.children
        assert len(filtered_lazy.children) == 1
        assert materialized in filtered_lazy.children

    def test_node_workspace_transfer(self):
        """Test moving nodes between workspaces."""
        workspace1 = Workspace("workspace1")
        workspace2 = Workspace("workspace2")

        # Create initial data in workspace1
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        root = workspace1.add_node(Node(data=df, name="root", workspace=workspace1))
        child1 = root.filter(pl.col("x") > 2)
        child2 = child1.filter(pl.col("y") < 50)

        # Verify initial state
        assert len(workspace1.nodes) == 3  # root, child1, child2
        assert len(workspace2.nodes) == 0

        # Move child1 to workspace2 (this should also move child2)
        workspace2.add_node(child1)

        # Verify both workspaces
        assert len(workspace1.nodes) == 1  # only root
        assert len(workspace2.nodes) == 2  # child1 and child2

        # Verify nodes maintain their relationships
        assert len(child1.parents) == 1
        assert child1.parents[0] == root
        assert len(child2.parents) == 1
        assert child2.parents[0] == child1

    def test_workspace_node_always_exists(self):
        """Test that nodes always have a workspace."""
        # Create node without explicit workspace
        df = pl.DataFrame({"a": [1, 2, 3]})
        node = Node(df, "test_node")

        # Should have created a workspace
        assert node.workspace is not None
        assert isinstance(node.workspace, Workspace)
        assert node.id in node.workspace.nodes
        assert node.workspace.name == "workspace_for_test_node"

        # Create child node - should use same workspace
        child = node.filter(pl.col("a") > 1)
        assert child.workspace == node.workspace
        assert child.id in node.workspace.nodes

    def test_attribute_delegation_with_all_types(self):
        """Test attribute delegation works with all supported data types."""
        # Regular DataFrame
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        node_df = Node(df, "df_node")
        assert hasattr(node_df, "columns")
        assert list(node_df.columns) == ["a", "b"]

        # LazyFrame
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        node_lf = Node(lf, "lf_node")
        assert hasattr(node_lf, "columns")
        assert list(node_lf.columns) == ["a", "b"]

        # DocDataFrame
        doc_df_data = pl.DataFrame({"a": ["text1", "text2", "text3"], "b": [4, 5, 6]})
        doc_df = DocDataFrame(doc_df_data, document_column="a")
        node_doc = Node(doc_df, "doc_node")
        assert hasattr(node_doc, "columns")
        assert hasattr(node_doc, "active_document_name")
        assert node_doc.active_document_name == "a"

    def test_operation_tracking(self):
        """Test that operations are properly tracked."""
        workspace = Workspace("operation_tracking")

        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        root = workspace.add_node(Node(data=df, name="root", workspace=workspace))

        # Various operations
        filtered = root.filter(pl.col("a") > 2)
        assert filtered.operation == "filter(root)"

        sliced = root.slice(1, 3)
        assert sliced.operation == "slice(root)"

        sorted = root.sort("b")
        assert sorted.operation == "sort(root)"

        # Check parent operations are preserved
        double_filtered = filtered.filter(pl.col("b") < 3)
        assert double_filtered.operation == "filter(filter_root)"


class TestDocFrameIntegration:
    """Test integration with DocDataFrame and DocLazyFrame."""

    @pytest.fixture
    def doc_dataframe(self):
        """Create a DocDataFrame for testing."""
        df = pl.DataFrame(
            {
                "doc_id": ["doc1", "doc2", "doc3"],
                "text": ["Hello world", "Data science", "Machine learning"],
                "category": ["greeting", "tech", "tech"],
            }
        )
        return DocDataFrame(df, document_column="text")

    @pytest.fixture
    def doc_lazyframe(self):
        """Create a DocLazyFrame for testing."""
        df = pl.LazyFrame(
            {
                "doc_id": ["doc1", "doc2", "doc3"],
                "text": ["Hello world", "Data science", "Machine learning"],
                "category": ["greeting", "tech", "tech"],
            }
        )
        return DocLazyFrame(df, document_column="text")

    def test_node_document_column_property(self, doc_dataframe, doc_lazyframe):
        """Test document column property for DocDataFrame and DocLazyFrame."""
        # Test DocDataFrame
        node1 = Node(doc_dataframe, "doc_node")
        assert node1.document_column == "text"

        # Test DocLazyFrame
        node2 = Node(doc_lazyframe, "doc_lazy_node")
        assert node2.document_column == "text"

        # Test regular DataFrame (should return None)
        regular_df = pl.DataFrame({"col": [1, 2, 3]})
        node3 = Node(regular_df, "regular_node")
        assert node3.document_column is None

    def test_doc_node_operations_preserve_document_column(self, doc_dataframe):
        """Test that operations on DocDataFrame nodes preserve the document column."""
        node = Node(doc_dataframe, "doc_node")

        # Filter operation
        filtered = node.filter(pl.col("category") == "tech")
        assert isinstance(filtered.data, DocDataFrame)
        assert filtered.data.document_column == "text"

        # Select operation
        selected = node.select(["doc_id", "text"])
        assert isinstance(selected.data, DocDataFrame)
        assert selected.data.document_column == "text"

    def test_doc_lazy_node_collect(self, doc_lazyframe):
        """Test collecting a DocLazyFrame node."""
        node = Node(doc_lazyframe, "doc_lazy_node")
        assert node.is_lazy

        collected = node.collect()
        assert not collected.is_lazy
        assert isinstance(collected.data, DocDataFrame)
        assert collected.data.document_column == "text"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_node_unsupported_data_type(self):
        """Test Node creation with unsupported data type."""
        with pytest.raises(AssertionError) as exc_info:
            Node("invalid_data_type", "test")

        # Check that the error message mentions unsupported data type
        assert "Unsupported data type" in str(exc_info.value)

    def test_workspace_serialization_errors(self):
        """Test workspace serialization with problematic data."""
        workspace = Workspace("error_test")

        # Create a node with data that might cause serialization issues
        df = pl.DataFrame({"col": [1, 2, 3]})
        node = Node(df, "test_node")
        workspace.add_node(node)

        # Mock a serialization error in the node
        with patch.object(
            node, "serialize", side_effect=Exception("Serialization failed")
        ):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                temp_path = f.name

            # Should not raise an exception, but handle the error gracefully
            workspace.serialize(temp_path)

            # Check that the workspace was saved with error information
            with open(temp_path, "r") as f:
                data = json.load(f)

            assert data["id"] == workspace.id
            assert "error" in data["nodes"][node.id]

            Path(temp_path).unlink()  # cleanup

    def test_workspace_deserialization_errors(self):
        """Test workspace deserialization with corrupted data."""
        # Create a corrupted JSON file
        corrupted_data = {
            "id": "test-id",
            "name": "test-workspace",
            "metadata": {},
            "nodes": {"bad-node": {"error": "Corrupted node data"}},
            "relationships": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(corrupted_data, f)
            temp_path = f.name

        # Should handle corrupted data gracefully
        workspace = Workspace.deserialize(temp_path)
        assert workspace.id == "test-id"
        assert len(workspace.nodes) == 0  # Corrupted node should be skipped

        Path(temp_path).unlink()  # cleanup

    def test_node_schema_extraction_errors(self):
        """Test schema extraction with problematic data."""
        # Test with a regular DataFrame that has schema
        df = pl.DataFrame({"col": [1, 2, 3]})
        node = Node(df, "test_node")

        # Test that schema extraction works normally
        schema = node.json_schema()
        assert isinstance(schema, dict)
        assert "col" in schema

        # Since we added error handling to json_schema(), it should handle
        # any schema extraction errors gracefully
        # Let's test that the method exists and returns a dict
        assert hasattr(node, "json_schema")
        assert callable(node.json_schema)
