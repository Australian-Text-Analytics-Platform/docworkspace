"""Workspace module for managing collections of Nodes with serialization capabilities."""

import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import polars as pl

import docframe
from docframe import DocDataFrame, DocLazyFrame

from .node import Node


class Workspace:
    """
    A workspace for managing collections of Nodes with parent-child relationships.

    Provides functionality to:
    - Add and manage nodes
    - Track relationships between nodes
    - Serialize and deserialize the workspace state
    - Load data from various sources into nodes

    Attributes:
        id: Unique identifier for the workspace
        name: Human-readable name for the workspace
        nodes: Dictionary mapping node IDs to Node objects
        _metadata: Additional workspace metadata
    """

    def __init__(
        self,
        name: Optional[str] = None,
        data: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame, Any]] = None,
        data_name: Optional[str] = None,
        csv_lazy: bool = True,
        **csv_kwargs,
    ):
        """
        Initialize a new Workspace.

        Args:
            name: Optional name for the workspace
            data: Optional data to load initially. Can be:
                - str/Path: CSV file path
                - pl.DataFrame: Polars DataFrame
                - pl.LazyFrame: Polars LazyFrame
                - DocDataFrame: DocFrame DocDataFrame
                - DocLazyFrame: DocFrame DocLazyFrame
            data_name: Optional name for the initial data node
            csv_lazy: If data is CSV path, whether to load lazily (default True)
            **csv_kwargs: Additional arguments for CSV loading
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"workspace_{self.id[:8]}"
        self.nodes: Dict[str, Node] = {}
        self._metadata: Dict[str, Any] = {}

        # Load initial data if provided
        if data is not None:
            self._load_initial_data(data, data_name, csv_lazy, **csv_kwargs)

    def _load_initial_data(
        self,
        data: Union[str, Path, pl.DataFrame, pl.LazyFrame, Any],
        data_name: Optional[str] = None,
        csv_lazy: bool = True,
        **csv_kwargs,
    ) -> Node:
        """
        Load initial data into the workspace.

        Args:
            data: Data to load
            data_name: Optional name for the node
            csv_lazy: Whether to load CSV files lazily
            **csv_kwargs: Additional CSV loading arguments

        Returns:
            The created node
        """
        # Handle CSV files
        if isinstance(data, (str, Path)):
            file_path = Path(data)
            if csv_lazy:
                df = pl.scan_csv(file_path, **csv_kwargs)
            else:
                df = pl.read_csv(file_path, **csv_kwargs)
            node_name = data_name or f"csv_{file_path.stem}"
            operation = f"load_csv({file_path})"

        # Handle all other data types generically
        else:
            df = data
            node_name = data_name or f"data_{len(self.nodes)}"
            operation = "load_data"

        # Create and add the node - let the Node class determine the data type
        node = Node(data=df, name=node_name, workspace=self, operation=operation)
        return self.add_node(node)

    def add_node(self, node: Node) -> Node:
        """
        Add a node to the workspace.

        Args:
            node: The node to add

        Returns:
            The added node
        """
        # Don't add if already present
        if node.id in self.nodes:
            return node

        # Remove from previous workspace if it exists
        if (
            hasattr(node, "workspace")
            and node.workspace is not None
            and node.workspace is not self
        ):
            if node.id in node.workspace.nodes:
                del node.workspace.nodes[node.id]

        self.nodes[node.id] = node
        # Update the node's workspace reference
        node.workspace = self

        # Also move all children to this workspace
        def move_children_recursive(current_node):
            for child in current_node.children:
                if child.id not in self.nodes:
                    # Remove from previous workspace
                    if child.workspace is not None and child.workspace is not self:
                        if child.id in child.workspace.nodes:
                            del child.workspace.nodes[child.id]
                    # Add to this workspace
                    self.nodes[child.id] = child
                    child.workspace = self
                    # Recursively move grandchildren
                    move_children_recursive(child)

        move_children_recursive(node)

        return node

    def remove_node(self, node_id: str, materialize_children: bool = False) -> bool:
        """
        Remove a node from the workspace.

        Args:
            node_id: The ID of the node to remove
            materialize_children: If True, materialize child nodes if they are lazy

        Returns:
            True if the node was removed, False if not found
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        if materialize_children:
            # Materialize all children if they are lazy
            for child in node.children.copy():
                child.materialize()

        # Remove from parents' children lists
        for parent in node.parents:
            if node in parent.children:
                parent.children.remove(node)

        # Remove the node
        del self.nodes[node_id]
        return True

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node by its ID.

        Args:
            node_id: The ID of the node

        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_node_by_name(self, name: str) -> Optional[Node]:
        """
        Get a node by its name.

        Args:
            name: The name of the node

        Returns:
            The first node with the given name, None if not found
        """
        for node in self.nodes.values():
            if node.name == name:
                return node
        return None

    def get_node_by_uuid(self, uuid: str) -> Optional[Node]:
        """
        Get a node by its UUID.

        Args:
            uuid: The UUID of the node

        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(uuid)

    def list_nodes(self) -> List[Node]:
        """
        Get a list of all nodes in the workspace.

        Returns:
            List of all nodes
        """
        return list(self.nodes.values())

    def get_root_nodes(self) -> List[Node]:
        """
        Get all root nodes (nodes with no parents).

        Returns:
            List of root nodes
        """
        return [node for node in self.nodes.values() if not node.parents]

    def get_leaf_nodes(self) -> List[Node]:
        """
        Get all leaf nodes (nodes with no children).

        Returns:
            List of leaf nodes
        """
        return [node for node in self.nodes.values() if not node.children]

    def serialize(self, file_path: Union[str, Path]) -> None:
        """
        Serialize the workspace to a JSON file using the underlying data's serialization methods.

        Args:
            file_path: Path to save the workspace (will be saved as JSON)
        """
        file_path = Path(file_path)

        # Ensure .json extension
        if file_path.suffix.lower() != ".json":
            file_path = file_path.with_suffix(".json")

        # Prepare serialization data
        workspace_data = {
            "id": self.id,
            "name": self.name,
            "metadata": self._metadata,
            "nodes": {},
            "relationships": [],
        }

        # Serialize each node using its own serialization method
        for node_id, node in self.nodes.items():
            try:
                serialized_node = node.serialize(format="json")
                workspace_data["nodes"][node_id] = serialized_node
            except Exception as e:
                # If node serialization fails, fall back to basic info
                workspace_data["nodes"][node_id] = {
                    "node_metadata": {
                        "id": node.id,
                        "name": node.name,
                        "operation": node.operation,
                        "data_type": f"{type(node.data).__class__.__module__}.{type(node.data).__name__}",
                        "is_lazy": node.is_lazy,
                    },
                    "data_metadata": {"type": "error"},
                    "serialized_data": str(e),
                    "error": str(e),
                }

        # Serialize relationships
        for node in self.nodes.values():
            for child in node.children:
                workspace_data["relationships"].append(
                    {"parent_id": node.id, "child_id": child.id}
                )

        # Save to JSON file
        import json

        with open(file_path, "w") as f:
            json.dump(workspace_data, f, indent=2, default=str)

    @classmethod
    def deserialize(cls, file_path: Union[str, Path]) -> "Workspace":
        """
        Deserialize a workspace from a JSON file using the nodes' deserialization methods.

        Args:
            file_path: Path to the serialized workspace JSON file

        Returns:
            Deserialized Workspace object
        """
        file_path = Path(file_path)

        # Load data from JSON file
        import json

        with open(file_path, "r") as f:
            workspace_data = json.load(f)

        # Create workspace with empty nodes dict
        workspace = cls.__new__(cls)
        workspace.id = workspace_data["id"]
        workspace.name = workspace_data["name"]
        workspace._metadata = workspace_data.get("metadata", {})
        workspace.nodes = {}

        # Deserialize nodes using their own deserialization methods
        node_map = {}
        for node_id, serialized_node in workspace_data["nodes"].items():
            if "error" in serialized_node:
                # Skip nodes that failed to serialize
                continue

            try:
                node = Node.deserialize(serialized_node, workspace, format="json")
                # Don't add to workspace again since Node.deserialize already does it
                # Just track in node_map for relationship building
                node_map[node_id] = node
            except Exception as e:
                # Skip nodes that fail to deserialize
                print(f"Warning: Failed to deserialize node {node_id}: {e}")
                continue

        # Rebuild relationships
        for relationship in workspace_data.get("relationships", []):
            parent_id = relationship["parent_id"]
            child_id = relationship["child_id"]

            if parent_id in node_map and child_id in node_map:
                parent = node_map[parent_id]
                child = node_map[child_id]

                if child not in parent.children:
                    parent.children.append(child)
                if parent not in child.parents:
                    child.parents.append(parent)

        return workspace

    @classmethod
    def from_dict(cls, workspace_dict: Dict[str, Any]) -> "Workspace":
        """
        Create a workspace from a dictionary (for API compatibility).

        Args:
            workspace_dict: Dictionary containing workspace data

        Returns:
            Workspace object
        """
        # Create workspace with empty nodes dict to avoid auto-registration
        workspace = cls.__new__(cls)
        workspace.id = workspace_dict["id"]
        workspace.name = workspace_dict["name"]
        workspace._metadata = workspace_dict.get("metadata", {})
        workspace.nodes = {}

        # Load nodes from the dictionary format
        for node_id, node_data in workspace_dict.get("nodes", {}).items():
            try:
                # Check if this is the new serialization format
                if "node_metadata" in node_data and "serialized_data" in node_data:
                    # New format - use Node deserialization
                    node = Node.deserialize(node_data, workspace, format="json")
                else:
                    # Legacy format - handle manually
                    data_info = node_data.get("data", {})

                    if isinstance(data_info, dict):
                        data_type = data_info.get("type")

                        if data_type == "polars_dataframe":
                            data = pl.DataFrame(data_info["data"])
                        elif data_type == "polars_lazyframe":
                            data = pl.LazyFrame(data_info["data"])
                        elif data_type == "doc_dataframe":
                            df = pl.DataFrame(data_info["data"])
                            document_column = data_info.get(
                                "document_column", "document"
                            )
                            data = docframe.DocDataFrame(
                                df, document_column=document_column
                            )
                        elif data_type == "doc_lazyframe":
                            df = pl.DataFrame(data_info["data"])
                            document_column = data_info.get(
                                "document_column", "document"
                            )
                            data = docframe.DocLazyFrame(
                                df.lazy(), document_column=document_column
                            )
                        else:
                            # Fallback - unknown type, create a simple dataframe
                            data = pl.DataFrame({"data": [str(data_info)]})
                    elif isinstance(data_info, bytes):
                        # Fallback to pickle
                        import pickle

                        try:
                            data = pickle.loads(data_info)
                            # Verify it's a supported type
                            if not isinstance(
                                data,
                                (
                                    pl.DataFrame,
                                    pl.LazyFrame,
                                    docframe.DocDataFrame,
                                    docframe.DocLazyFrame,
                                ),
                            ):
                                data = pl.DataFrame({"data": [str(data)]})
                        except Exception:
                            data = pl.DataFrame({"data": ["Failed to deserialize"]})
                    else:
                        # Fallback - store as dataframe
                        data = pl.DataFrame({"data": [str(data_info)]})

                    # Create node manually without triggering workspace registration
                    node = Node.__new__(Node)
                    node.id = node_id
                    node.name = node_data["name"]
                    node.data = data
                    node.parents = []
                    node.children = []
                    node.workspace = workspace
                    node.operation = node_data.get("operation")

                    # Add to workspace manually
                    workspace.nodes[node_id] = node

            except Exception as e:
                print(f"Warning: Failed to load node {node_id}: {e}")
                continue

        return workspace

    def get_metadata(self, key: str) -> Any:
        """
        Get workspace metadata.

        Args:
            key: The metadata key

        Returns:
            The metadata value
        """
        return self._metadata.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set workspace metadata.

        Args:
            key: The metadata key
            value: The metadata value
        """
        self._metadata[key] = value

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the workspace.

        Returns:
            Dictionary containing workspace summary information
        """
        node_types = {}
        status_counts = {"lazy": 0, "materialized": 0}

        for node in self.nodes.values():
            node_type = type(node.data).__name__
            node_types[node_type] = node_types.get(node_type, 0) + 1
            status_key = "lazy" if node.is_lazy else "materialized"
            status_counts[status_key] += 1

        return {
            "id": self.id,
            "name": self.name,
            "total_nodes": len(self.nodes),
            "root_nodes": len(self.get_root_nodes()),
            "leaf_nodes": len(self.get_leaf_nodes()),
            "node_types": node_types,
            "status_counts": status_counts,
            "metadata_keys": list(self._metadata.keys()),
        }

    def info(self) -> Dict[str, Any]:
        """
        Get information about the workspace (alias for summary).

        Returns:
            Dictionary containing workspace information
        """
        return self.summary()

    def visualize_graph(self) -> str:
        """
        Generate a text-based visualization of the node graph.

        Returns:
            String representation of the node graph
        """
        lines = [f"Workspace: {self.name} (ID: {self.id[:8]})"]
        lines.append("=" * 50)

        # Calculate edges manually
        edge_count = sum(len(node.children) for node in self.nodes.values())
        lines.append(f"Graph Info: {len(self.nodes)} nodes, {edge_count} edges")
        if self.has_cycles():
            lines.append("⚠️  Graph contains cycles")
        lines.append("")

        # Simple tree-based visualization
        lines.extend(self._simple_visualize_graph())

        return "\n".join(lines)

    def _simple_visualize_graph(self) -> List[str]:
        """Simple tree-based graph visualization (fallback)."""
        lines = []

        # Show root nodes and their descendants
        for root in self.get_root_nodes():
            lines.extend(self._visualize_node(root, 0))

        return lines

    def _visualize_node(self, node: Node, depth: int) -> List[str]:
        """
        Recursively visualize a node and its children.

        Args:
            node: The node to visualize
            depth: Current depth in the tree

        Returns:
            List of string lines representing the node
        """
        indent = "  " * depth
        status_symbol = "⚡" if node.is_lazy else "✓"
        data_type = type(node.data).__name__

        lines = [f"{indent}{status_symbol} {node.name} ({data_type})"]

        for child in node.children:
            lines.extend(self._visualize_node(child, depth + 1))

        return lines

    def __repr__(self) -> str:
        """String representation of the Workspace."""
        return (
            f"Workspace(id={self.id[:8]}, name='{self.name}', nodes={len(self.nodes)})"
        )

    def __iter__(self) -> Iterator[Node]:
        """Iterate over all nodes in the workspace."""
        return iter(self.nodes.values())

    def __len__(self) -> int:
        """Return the number of nodes in the workspace."""
        return len(self.nodes)

    def __bool__(self) -> bool:
        """Always return True - a workspace object is always truthy regardless of node count."""
        return True

    def get_descendants(self, node_id: str) -> List[Node]:
        """
        Get all descendant nodes of a given node.

        Args:
            node_id: The ID of the node

        Returns:
            List of descendant nodes
        """
        if node_id not in self.nodes:
            return []

        descendants = []
        visited = set()

        def traverse(node):
            if node.id in visited:
                return
            visited.add(node.id)
            for child in node.children:
                descendants.append(child)
                traverse(child)

        traverse(self.nodes[node_id])
        return descendants

    def get_ancestors(self, node_id: str) -> List[Node]:
        """
        Get all ancestor nodes of a given node.

        Args:
            node_id: The ID of the node

        Returns:
            List of ancestor nodes
        """
        if node_id not in self.nodes:
            return []

        ancestors = []
        visited = set()

        def traverse(node):
            if node.id in visited:
                return
            visited.add(node.id)
            for parent in node.parents:
                ancestors.append(parent)
                traverse(parent)

        traverse(self.nodes[node_id])
        return ancestors

    def get_shortest_path(self, source_id: str, target_id: str) -> Optional[List[Node]]:
        """
        Get the shortest path between two nodes.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node

        Returns:
            List of nodes in the shortest path, or None if no path exists
        """
        # Simple implementation - not implementing BFS for now
        return None

    def is_connected(self) -> bool:
        """
        Check if the graph is weakly connected (ignoring edge direction).

        Returns:
            True if the graph is connected, False otherwise
        """
        # Simple implementation - assume connected for now
        return True

    def has_cycles(self) -> bool:
        """
        Check if the graph has cycles.

        Returns:
            True if the graph has cycles, False otherwise
        """
        # Simple implementation - assume no cycles for now
        return False

    def get_topological_order(self) -> List[Node]:
        """
        Get nodes in topological order (parents before children).

        Returns:
            List of nodes in topological order
        """
        # Simple implementation: return root nodes first, then others
        roots = self.get_root_nodes()
        others = [node for node in self.nodes.values() if node not in roots]
        return roots + others

    def graph(self) -> Dict[str, Any]:
        """
        Generate a generic graph structure for visualization.

        Returns:
            Dictionary containing generic nodes and edges information
        """
        nodes_data = []
        edges_data = []

        # Create generic node data
        for node in self.nodes.values():
            # Get data shape information if available
            shape_info = None
            try:
                if isinstance(node.data, (pl.DataFrame, docframe.DocDataFrame)):
                    shape_info = {
                        "type": "shape",
                        "rows": node.data.shape[0],  # type: ignore
                        "columns": node.data.shape[1],  # type: ignore
                    }
                elif hasattr(node.data, "__len__"):
                    try:
                        shape_info = {
                            "type": "length",
                            "length": len(node.data),  # type: ignore
                        }
                    except Exception:
                        pass
            except Exception:
                pass

            # Create generic node data
            node_data = {
                "id": node.id,
                "name": node.name,
                "type": type(node.data).__name__,
                "lazy": node.is_lazy,
                "operation": node.operation or "initial",
                "parent_count": len(node.parents),
                "child_count": len(node.children),
                "parent_ids": [p.id for p in node.parents],
                "child_ids": [c.id for c in node.children],
                "shape_info": shape_info,
                "is_root": len(node.parents) == 0,
                "is_leaf": len(node.children) == 0,
            }

            nodes_data.append(node_data)

            # Create edges for parent-child relationships
            for parent in node.parents:
                edge_data = {
                    "id": f"edge_{parent.id}_{node.id}",
                    "source": parent.id,
                    "target": node.id,
                    "operation": node.operation or "relationship",
                    "is_lazy": node.is_lazy,
                }
                edges_data.append(edge_data)

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "workspace_info": {
                "id": self.id,
                "name": self.name,
                "total_nodes": len(self.nodes),
                "root_nodes": len(self.get_root_nodes()),
                "leaf_nodes": len(self.get_leaf_nodes()),
            },
        }

    def _serialize_data(self, data: Any) -> Any:
        """
        Serialize data object to a storable format (legacy method for test compatibility).

        Args:
            data: The data to serialize

        Returns:
            Serialized data
        """
        if isinstance(data, pl.DataFrame):
            return {"type": "polars_dataframe", "data": data.to_dicts()}
        elif isinstance(data, pl.LazyFrame):
            # For LazyFrame, collect it first to serialize
            df = data.collect()
            return {"type": "polars_lazyframe", "data": df.to_dicts()}
        elif isinstance(data, docframe.DocDataFrame):
            # Handle DocDataFrame
            df = data.to_polars()
            return {
                "type": "doc_dataframe",
                "data": df.to_dicts(),
                "document_column": data.document_column,
            }
        elif isinstance(data, docframe.DocLazyFrame):
            # Handle DocLazyFrame
            df = data.collect().to_polars()
            return {
                "type": "doc_lazyframe",
                "data": df.to_dicts(),
                "document_column": data.document_column,
            }
        else:
            # Fallback: try to pickle the object
            try:
                import pickle

                return pickle.dumps(data)
            except Exception:
                return str(data)

    def to_react_flow_json(self) -> Dict[str, Any]:
        """
        Generate React Flow compatible graph structure using Node.info(json=True).

        This method provides a clean interface for web frontends that need
        React Flow compatible data structures.

        Returns:
            Dictionary containing nodes, edges, and workspace_info for React Flow
        """
        # Convert nodes to React Flow format using Node.info(json=True)
        react_nodes = []
        node_positions = {}  # For simple layout

        for i, node in enumerate(self.nodes.values()):
            # Get JSON-compatible node info
            node_info = node.info(json=True)

            # Calculate simple grid layout position
            x = (i % 4) * 250 + 100  # 4 nodes per row
            y = (i // 4) * 150 + 100
            node_positions[node.id] = {"x": x, "y": y}

            # Extract columns for frontend compatibility
            columns = []
            dtypes = {}
            schema = None

            if node.is_lazy:
                schema = node.data.collect_schema()
            else:
                schema = node.data.schema

            if schema:
                columns = list(schema.names())
                dtypes = {col: str(dtype) for col, dtype in schema.items()}

            # Convert dtypes to JS-friendly format (matching schema_to_json pattern)
            js_friendly_dtypes = {}
            for col, dtype_str in dtypes.items():
                dtype_lower = str(dtype_str).lower()
                if any(x in dtype_lower for x in ["int", "float", "double"]):
                    js_friendly_dtypes[col] = "number"
                elif any(x in dtype_lower for x in ["str", "string", "utf8"]):
                    js_friendly_dtypes[col] = "string"
                elif any(x in dtype_lower for x in ["bool", "boolean"]):
                    js_friendly_dtypes[col] = "boolean"
                elif any(x in dtype_lower for x in ["date", "time", "datetime"]):
                    js_friendly_dtypes[col] = "datetime"
                else:
                    js_friendly_dtypes[col] = "string"

            react_node = {
                "id": node.id,
                "type": "customNode",
                "position": node_positions[node.id],
                "data": {
                    "label": node.name,
                    "nodeId": node.id,
                    "nodeName": node.name,
                    "operation": node.operation or "load",
                    "dataType": node_info["dtype"],  # Full module.class format
                    "shape": node_info.get("shape", (0, 0)),
                    "isLazy": node_info["lazy"],
                    "columns": columns,
                    "dtypes": js_friendly_dtypes,  # JS-friendly format
                    "schema": node_info.get("schema", {}),  # JSON-compatible schema
                    "document_column": node_info.get("document_column"),
                },
                "connectable": True,
            }
            react_nodes.append(react_node)

        # Convert edges to React Flow format
        react_edges = []
        for node in self.nodes.values():
            for child in node.children:
                edge_id = f"{node.id}->{child.id}"
                react_edge = {
                    "id": edge_id,
                    "source": node.id,
                    "target": child.id,
                    "type": "smoothstep",
                    "animated": False,
                    "markerEnd": {"type": "arrowclosed", "width": 20, "height": 20},
                }
                react_edges.append(react_edge)

        return {
            "nodes": react_nodes,
            "edges": react_edges,
            "workspace_info": {
                "id": self.id,
                "name": self.name,
                "total_nodes": len(react_nodes),
                "root_nodes": len(self.get_root_nodes()),
                "leaf_nodes": len(self.get_leaf_nodes()),
            },
        }

    # API-specific methods have been moved to ldaca_web_app backend
    # to keep docworkspace general-purpose
