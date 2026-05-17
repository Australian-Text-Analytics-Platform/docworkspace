"""Workspace analysis helpers.

`info_json` returns concise structural workspace metrics.
`graph_json` returns graph-only payloads (nodes + edges).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    from .core import Workspace


def info_json(workspace: "Workspace") -> Dict[str, Any]:
    total_nodes = len(workspace.nodes)
    root_nodes = len(workspace.get_root_nodes())
    leaf_nodes = len(workspace.get_leaf_nodes())

    return {
        "name": workspace.name,
        "id": workspace.id,
        "description": workspace.description or "",
        "created_at": workspace.created_at,
        "modified_at": workspace.modified_at,
        "total_nodes": total_nodes,
        "root_nodes": root_nodes,
        "leaf_nodes": leaf_nodes,
    }


def graph_json(workspace: "Workspace") -> Dict[str, object]:
    nodes_payload: List[Dict[str, object]] = []
    edges_payload: List[Dict[str, str]] = []

    for node in workspace.nodes.values():
        try:
            nodes_payload.append(node.info())
        except Exception as exc:
            # Per-node fallback: one broken node (e.g. missing source file,
            # undeserializable lazy plan) must not take down the whole graph.
            nodes_payload.append(
                {
                    "id": node.id,
                    "name": getattr(node, "name", node.id),
                    "operation": getattr(node, "operation", "unknown"),
                    "child_ids": [c.id for c in getattr(node, "children", [])],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        for child in node.children:
            edges_payload.append({"source": node.id, "target": child.id})

    return {
        "nodes": nodes_payload,
        "edges": edges_payload,
    }


__all__ = ["info_json", "graph_json"]
