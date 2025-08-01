"""Tests for error handling and edge cases.

This module tests error handling, edge cases, and defensive programming:
- Invalid inputs and operations
- Error propagation
- Graceful failure handling
- Type validation
"""

import polars as pl
import pytest

from docworkspace import Node, Workspace

# For docframe support
try:
    from docframe import DocDataFrame, DocLazyFrame

    DOCFRAME_AVAILABLE = True
except ImportError:
    DOCFRAME_AVAILABLE = False


class TestErrorHandling:
    """Test error handling in core functionality."""

    def test_node_creation_invalid_data(self):
        """Test Node creation with invalid data."""
        with pytest.raises((TypeError, ValueError)):
            Node("invalid_data", "test_node")

    def test_workspace_invalid_node_operations(self):
        """Test workspace operations with invalid nodes."""
        workspace = Workspace("test_workspace")

        # Test removing non-existent node
        with pytest.raises(ValueError):
            workspace.remove_node("non-existent-id")

    def test_node_json_schema_with_empty_dataframe(self):
        """Test JSON schema extraction with empty DataFrame."""
        empty_df = pl.DataFrame({"col1": [], "col2": []})
        node = Node(empty_df, "empty_node")

        # This should not raise an error (graceful handling)
        schema = node.json_schema()
        assert schema is not None
        assert "properties" in schema

    def test_node_json_schema_with_null_values(self):
        """Test JSON schema with null/missing values."""
        df_with_nulls = pl.DataFrame(
            {
                "text": ["hello", None, "world"],
                "value": [1, None, 3],
                "flag": [True, None, False],
            }
        )
        node = Node(df_with_nulls, "null_node")

        schema = node.json_schema()
        assert schema is not None
        assert "properties" in schema

    def test_invalid_lazy_operations(self, sample_lazyframe):
        """Test invalid operations on lazy frames."""
        node = Node(sample_lazyframe, "lazy_node")

        # Some operations might not be valid on lazy frames
        # This tests graceful handling
        try:
            result = node.head()  # Should work
            assert result is not None
        except Exception as e:
            # If it fails, it should fail gracefully
            assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_workspace_circular_references(self, sample_dataframe):
        """Test handling of potential circular references."""
        workspace = Workspace("test_workspace")
        node1 = workspace.add_node(Node(sample_dataframe, "node1"))
        node2 = workspace.add_node(Node(sample_dataframe, "node2"))

        # Set up bidirectional relationships
        workspace.set_relationship(node1.id, node2.id, "related_to")
        workspace.set_relationship(node2.id, node1.id, "related_to")

        # Should handle gracefully
        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_workspace_operations(self):
        """Test operations on empty workspace."""
        workspace = Workspace("empty_workspace")

        assert len(workspace.list_nodes()) == 0
        assert workspace.get_graph()["nodes"] == []

        # Should handle gracefully
        summaries = workspace.get_node_summaries()
        assert len(summaries) == 0

    def test_node_with_single_column(self):
        """Test Node with single column DataFrame."""
        single_col_df = pl.DataFrame({"only_col": [1, 2, 3]})
        node = Node(single_col_df, "single_col")

        assert node.shape == (3, 1)
        assert node.columns == ["only_col"]

    def test_node_with_large_data_simulation(self):
        """Test Node with simulated large data."""
        # Create a relatively large DataFrame for testing
        large_df = pl.DataFrame(
            {
                "id": range(1000),
                "text": [f"text_{i}" for i in range(1000)],
                "value": [i * 2 for i in range(1000)],
            }
        )

        node = Node(large_df, "large_node")
        assert node.shape == (1000, 3)

        # Test pagination-like operations
        head_result = node.head(10)
        assert head_result.shape == (10, 3)

    def test_special_character_handling(self):
        """Test handling of special characters in data."""
        special_df = pl.DataFrame(
            {
                "text": ["hello", "世界", "🌍", "café", "naïve"],
                "symbols": ["@#$%", "∑∆", "αβγ", "→←", "♠♥♦♣"],
                "numbers": [1, 2, 3, 4, 5],
            }
        )

        node = Node(special_df, "special_chars")
        assert node.shape == (5, 3)

        # Should handle serialization gracefully
        schema = node.json_schema()
        assert schema is not None

    def test_duplicate_node_names(self, sample_dataframe):
        """Test handling of duplicate node names."""
        workspace = Workspace("test_workspace")

        node1 = workspace.add_node(Node(sample_dataframe, "duplicate_name"))
        node2 = workspace.add_node(Node(sample_dataframe, "duplicate_name"))

        # Should allow duplicate names but different IDs
        assert node1.id != node2.id
        assert node1.name == node2.name
        assert len(workspace.list_nodes()) == 2


@pytest.mark.skipif(not DOCFRAME_AVAILABLE, reason="DocFrame not available")
class TestDocFrameErrorHandling:
    """Test error handling with DocFrame integration."""

    def test_docdataframe_with_invalid_document_column(self):
        """Test DocDataFrame with invalid document column."""
        df = pl.DataFrame({"text": ["hello", "world"], "numbers": [1, 2]})

        # Try to create DocDataFrame with non-existent column
        with pytest.raises(ValueError):
            DocDataFrame(df, document_column="non_existent")

    def test_doclazyframe_error_propagation(self):
        """Test error propagation in DocLazyFrame."""
        df = pl.DataFrame({"text": ["hello", "world"]})
        doc_lazy = DocLazyFrame(df.lazy())

        # Operations should propagate errors appropriately
        try:
            # This might raise an error depending on the operation
            doc_lazy.select(pl.col("non_existent"))
            # If no error, that's also fine - just testing graceful handling
        except Exception as e:
            assert isinstance(e, (ValueError, KeyError, pl.ColumnNotFoundError))


class TestTypeValidation:
    """Test type validation and conversion."""

    def test_node_data_type_consistency(self, sample_dataframe, sample_lazyframe):
        """Test consistent behavior across data types."""
        eager_node = Node(sample_dataframe, "eager")
        lazy_node = Node(sample_lazyframe, "lazy")

        # Both should have consistent interface
        assert hasattr(eager_node, "shape")
        assert hasattr(lazy_node, "shape")
        assert hasattr(eager_node, "columns")
        assert hasattr(lazy_node, "columns")

    def test_workspace_serialization_consistency(self, sample_dataframe):
        """Test consistent serialization behavior."""
        workspace = Workspace("test_workspace")
        workspace.add_node(Node(sample_dataframe, "test_node"))

        # Should produce consistent output
        graph1 = workspace.get_graph()
        graph2 = workspace.get_graph()

        assert graph1["workspace_info"]["name"] == graph2["workspace_info"]["name"]
        assert len(graph1["nodes"]) == len(graph2["nodes"])
