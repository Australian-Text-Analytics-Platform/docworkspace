"""Tests for Workspace node-ordering helpers.

Covers ``reorder_nodes`` and ``place_node_after_parent``, the two methods that
back the workspace list-view drag-to-reorder gesture and smart child insertion.
"""

import polars as pl

from docworkspace import Node, Workspace


def _root(workspace: Workspace, name: str) -> Node:
    """Create and append a standalone root node for ordering tests."""
    return workspace.add_node(
        Node(data=pl.DataFrame({"x": [1, 2, 3]}).lazy(), name=name, workspace=workspace)
    )


class TestReorderNodes:
    def test_reorder_full_permutation(self):
        ws = Workspace("order_ws")
        a, b, c = _root(ws, "a"), _root(ws, "b"), _root(ws, "c")
        assert list(ws.nodes) == [a.id, b.id, c.id]

        result = ws.reorder_nodes([c.id, a.id, b.id])

        assert result == [c.id, a.id, b.id]
        assert list(ws.nodes) == [c.id, a.id, b.id]

    def test_reorder_ignores_unknown_ids(self):
        ws = Workspace("order_ws")
        a, b = _root(ws, "a"), _root(ws, "b")

        ws.reorder_nodes([b.id, "does-not-exist", a.id])

        assert list(ws.nodes) == [b.id, a.id]

    def test_reorder_appends_missing_nodes(self):
        ws = Workspace("order_ws")
        a, b, c = _root(ws, "a"), _root(ws, "b"), _root(ws, "c")

        # Only mention one id; the rest keep their relative order at the end.
        ws.reorder_nodes([c.id])

        assert list(ws.nodes) == [c.id, a.id, b.id]

    def test_reorder_ignores_duplicate_ids(self):
        ws = Workspace("order_ws")
        a, b = _root(ws, "a"), _root(ws, "b")

        ws.reorder_nodes([b.id, b.id, a.id])

        assert list(ws.nodes) == [b.id, a.id]


class TestPlaceNodeAfterParent:
    def test_places_child_directly_below_parent(self):
        ws = Workspace("order_ws")
        a = _root(ws, "a")
        b = _root(ws, "b")  # noqa: F841 - sits between parent and appended child
        c = _root(ws, "c")  # noqa: F841

        child = a.filter(pl.col("x") > 1)
        # Freshly created child is appended at the very end.
        assert list(ws.nodes)[-1] == child.id

        ws.place_node_after_parent(child)

        order = list(ws.nodes)
        assert order.index(child.id) == order.index(a.id) + 1

    def test_accepts_node_id(self):
        ws = Workspace("order_ws")
        a = _root(ws, "a")
        _root(ws, "b")
        child = a.filter(pl.col("x") > 1)

        ws.place_node_after_parent(child.id)

        order = list(ws.nodes)
        assert order.index(child.id) == order.index(a.id) + 1

    def test_root_node_without_parent_is_noop(self):
        ws = Workspace("order_ws")
        a = _root(ws, "a")
        b = _root(ws, "b")
        before = list(ws.nodes)

        ws.place_node_after_parent(b)

        assert list(ws.nodes) == before
        assert a.id in ws.nodes

    def test_unknown_node_is_noop(self):
        ws = Workspace("order_ws")
        _root(ws, "a")
        before = list(ws.nodes)

        ws.place_node_after_parent("missing-id")

        assert list(ws.nodes) == before


def test_saved_order_round_trips(tmp_path):
    """add_node must keep append order so a saved workspace reloads unchanged."""
    ws = Workspace("order_ws")
    a, b, c = _root(ws, "a"), _root(ws, "b"), _root(ws, "c")
    ws.reorder_nodes([c.id, b.id, a.id])

    meta_path = tmp_path / "metadata.json"
    ws.save(meta_path)
    loaded = Workspace.load(meta_path)

    assert list(loaded.nodes) == [c.id, b.id, a.id]
