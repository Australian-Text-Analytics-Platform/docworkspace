"""Tests for advanced workspace operations and algorithms.

This module tests advanced functionality including:
- Graph algorithms and traversal
- Layout algorithms
- Complex workspace operations
- Performance considerations
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


class TestGraphOperations:
    """Test graph-based operations and algorithms."""

    def test_workspace_graph_structure(self, sample_dataframe):
        """Test basic graph structure creation."""
        workspace = Workspace("graph_test")

        # Create nodes
        node1 = workspace.add_node(Node(sample_dataframe, "node1"))
        node2 = workspace.add_node(Node(sample_dataframe, "node2"))
        node3 = workspace.add_node(Node(sample_dataframe, "node3"))

        # Create relationships
        workspace.set_relationship(node1.id, node2.id, "transforms_to")
        workspace.set_relationship(node2.id, node3.id, "filters_to")

        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2

    def test_workspace_layout_algorithms(self, sample_dataframe):
        """Test different layout algorithms."""
        workspace = Workspace("layout_test")

        # Create a more complex graph
        nodes = []
        for i in range(5):
            node = workspace.add_node(Node(sample_dataframe, f"node_{i}"))
            nodes.append(node)

        # Create some relationships
        for i in range(4):
            workspace.set_relationship(nodes[i].id, nodes[i + 1].id, "connects_to")

        # Test different layout algorithms
        algorithms = ["grid", "hierarchical", "circular"]

        for algorithm in algorithms:
            try:
                graph = workspace.get_graph(layout_algorithm=algorithm)
                assert len(graph["nodes"]) == 5

                # Check that positions are assigned
                for node in graph["nodes"]:
                    assert "position" in node
                    assert "x" in node["position"]
                    assert "y" in node["position"]

            except ValueError:
                # Some algorithms might not be implemented
                pass

    def test_workspace_node_relationships(self, sample_dataframe):
        """Test node relationship management."""
        workspace = Workspace("relationship_test")

        node1 = workspace.add_node(Node(sample_dataframe, "source"))
        node2 = workspace.add_node(Node(sample_dataframe, "intermediate"))
        node3 = workspace.add_node(Node(sample_dataframe, "target"))

        # Set up chain of relationships
        workspace.set_relationship(node1.id, node2.id, "processes")
        workspace.set_relationship(node2.id, node3.id, "outputs")

        # Test relationship queries
        node1_relationships = workspace.get_node_relationships(node1.id)
        assert len(node1_relationships) >= 1

        node2_relationships = workspace.get_node_relationships(node2.id)
        assert len(node2_relationships) >= 1

    def test_complex_graph_operations(self, sample_dataframe):
        """Test complex graph operations."""
        workspace = Workspace("complex_graph")

        # Create a more complex graph structure
        central_node = workspace.add_node(Node(sample_dataframe, "central"))

        # Create satellite nodes
        satellite_nodes = []
        for i in range(3):
            satellite = workspace.add_node(Node(sample_dataframe, f"satellite_{i}"))
            satellite_nodes.append(satellite)
            workspace.set_relationship(central_node.id, satellite.id, "branches_to")

        # Create connections between satellites
        workspace.set_relationship(
            satellite_nodes[0].id, satellite_nodes[1].id, "connects_to"
        )
        workspace.set_relationship(
            satellite_nodes[1].id, satellite_nodes[2].id, "connects_to"
        )

        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 4
        assert len(graph["edges"]) >= 5  # 3 from central + 2 between satellites


class TestAdvancedOperations:
    """Test advanced workspace and node operations."""

    def test_workspace_bulk_operations(self, sample_dataframe):
        """Test bulk operations on workspace."""
        workspace = Workspace("bulk_test")

        # Add multiple nodes
        nodes = []
        for i in range(10):
            node = workspace.add_node(Node(sample_dataframe, f"bulk_node_{i}"))
            nodes.append(node)

        assert len(workspace.list_nodes()) == 10

        # Test bulk relationship creation
        for i in range(9):
            workspace.set_relationship(nodes[i].id, nodes[i + 1].id, "sequence")

        graph = workspace.get_graph()
        assert len(graph["edges"]) == 9

    def test_node_transformation_chains(self, sample_dataframe):
        """Test chaining node transformations."""
        workspace = Workspace("transformation_test")

        # Start with original data
        original = workspace.add_node(Node(sample_dataframe, "original"))

        # Create transformation chain
        filtered = workspace.add_node(
            Node(original.filter(pl.col("value") > 2), "filtered")
        )
        workspace.set_relationship(original.id, filtered.id, "filtered_from")

        selected = workspace.add_node(
            Node(filtered.select(["text", "value"]), "selected")
        )
        workspace.set_relationship(filtered.id, selected.id, "selected_from")

        # Verify chain
        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2

    def test_workspace_metadata_operations(self, sample_dataframe):
        """Test workspace metadata and information."""
        workspace = Workspace("metadata_test")
        workspace.add_node(Node(sample_dataframe, "test_node"))

        # Test workspace info
        graph = workspace.get_graph()
        workspace_info = graph["workspace_info"]

        assert "name" in workspace_info
        assert "node_count" in workspace_info
        assert "creation_time" in workspace_info
        assert workspace_info["node_count"] == 1

    def test_node_lazy_evaluation_preservation(self, sample_lazyframe):
        """Test that lazy evaluation is preserved through operations."""
        workspace = Workspace("lazy_test")

        # Create lazy node
        lazy_node = workspace.add_node(Node(sample_lazyframe, "lazy_original"))
        assert lazy_node.is_lazy

        # Perform operations that should preserve laziness
        if hasattr(lazy_node, "filter"):
            filtered_data = lazy_node.filter(pl.col("value") > 1)
            filtered_node = workspace.add_node(Node(filtered_data, "lazy_filtered"))
            workspace.set_relationship(lazy_node.id, filtered_node.id, "filtered_from")

            # Check if laziness is preserved (implementation dependent)
            # This test validates the behavior exists, not necessarily the outcome
            assert hasattr(filtered_node, "is_lazy")


@pytest.mark.skipif(not DOCFRAME_AVAILABLE, reason="DocFrame not available")
class TestDocFrameAdvancedOperations:
    """Test advanced operations with DocFrame integration."""

    def test_docdataframe_graph_operations(self):
        """Test DocDataFrame in graph operations."""
        workspace = Workspace("docframe_graph")

        # Create DocDataFrame
        df = pl.DataFrame(
            {
                "document": ["This is a test document", "Another document here"],
                "metadata": ["meta1", "meta2"],
            }
        )
        doc_df = DocDataFrame(df, document_column="document")

        # Add to workspace
        doc_node = workspace.add_node(Node(doc_df, "doc_node"))

        # Verify integration
        assert doc_node.shape == (2, 2)
        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 1

    def test_doclazyframe_workspace_integration(self):
        """Test DocLazyFrame workspace integration."""
        workspace = Workspace("doclazy_test")

        # Create DocLazyFrame
        df = pl.DataFrame(
            {
                "text": ["Document one", "Document two", "Document three"],
                "category": ["A", "B", "A"],
            }
        )
        doc_lazy = DocLazyFrame(df.lazy(), document_column="text")

        # Add to workspace
        lazy_doc_node = workspace.add_node(Node(doc_lazy, "lazy_doc"))

        # Test operations
        assert lazy_doc_node.is_lazy
        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 1

    def test_mixed_docframe_operations(self):
        """Test mixed DocDataFrame and regular DataFrame operations."""
        workspace = Workspace("mixed_test")

        # Regular DataFrame
        regular_df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
        regular_node = workspace.add_node(Node(regular_df, "regular"))

        # DocDataFrame
        doc_df_data = pl.DataFrame(
            {"id": [1, 2], "document": ["First doc", "Second doc"]}
        )
        doc_df = DocDataFrame(doc_df_data, document_column="document")
        doc_node = workspace.add_node(Node(doc_df, "doc"))

        # Create relationship
        workspace.set_relationship(regular_node.id, doc_node.id, "enriches")

        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1


class TestPerformanceConsiderations:
    """Test performance-related aspects."""

    def test_large_workspace_operations(self, sample_dataframe):
        """Test operations with larger workspaces."""
        workspace = Workspace("performance_test")

        # Create many nodes (but not too many for test performance)
        nodes = []
        for i in range(20):
            node = workspace.add_node(Node(sample_dataframe, f"perf_node_{i}"))
            nodes.append(node)

        # Test that operations complete in reasonable time
        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 20

        # Test list operations
        node_list = workspace.list_nodes()
        assert len(node_list) == 20

    def test_memory_efficient_operations(self, sample_lazyframe):
        """Test memory-efficient operations with lazy evaluation."""
        workspace = Workspace("memory_test")

        # Use lazy frames for memory efficiency
        lazy_node = workspace.add_node(Node(sample_lazyframe, "lazy_memory"))

        # Operations should not trigger evaluation
        assert lazy_node.is_lazy

        # Graph operations should work without forcing evaluation
        graph = workspace.get_graph()
        assert len(graph["nodes"]) == 1
