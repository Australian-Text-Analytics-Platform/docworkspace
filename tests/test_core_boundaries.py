"""Tests for the public core package boundary."""

import polars as pl

from docworkspace import Node, Workspace


def test_core_package_exports_only_workspace_types():
    from docworkspace import __all__

    assert set(__all__) == {
        "Node",
        "Workspace",
        "TokenizationMeta",
    }


def test_core_classes_do_not_expose_api_helpers():
    workspace = Workspace("test")
    node = workspace.add_node(
        Node(
            pl.DataFrame({"id": [1, 2, 3], "text": ["a", "b", "c"]}).lazy(), "test_node"
        )
    )

    api_methods = {
        "to_api_summary",
        "get_paginated_data",
        "to_api_graph",
        "get_node_summaries",
        "safe_operation",
        "set_relationship",
    }

    for method in api_methods:
        assert not hasattr(node, method)
        assert not hasattr(workspace, method)
