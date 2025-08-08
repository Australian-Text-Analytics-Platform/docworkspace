"""Node module for wrapping DataFrames and DocDataFrames with relationship tracking."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import polars as pl

from docframe import DocDataFrame, DocLazyFrame

if TYPE_CHECKING:
    from .workspace import Workspace

# Type alias for supported data types
SupportedDataTypes = pl.DataFrame | pl.LazyFrame | DocDataFrame | DocLazyFrame

# Supported data types (for documentation):
# - pl.DataFrame, pl.LazyFrame (polars)
# - DocDataFrame, DocLazyFrame (docframe)
# - Any other data type with compatible methods


def schema_to_json(schema: pl.Schema) -> Dict[str, str]:
    """
    Convert a Polars schema to a JSON-compatible dictionary.

    Args:
        schema: Polars Schema object

    Returns:
        Dictionary with column names as keys and their types as values
    """

    def python_type_to_str(type: Any) -> str:
        """
        Convert a Python type to its string representation for JSON compatibility.
        """
        match type:
            case (
                pl.Int8
                | pl.Int16
                | pl.Int32
                | pl.Int64
                | pl.UInt8
                | pl.UInt16
                | pl.UInt32
                | pl.UInt64
                | pl.Float32
                | pl.Float64
            ):
                return "number"
            case pl.String | pl.Utf8:  # Handle both String and Utf8 types
                return "string"
            case pl.Boolean:
                return "boolean"
            case pl.Date | pl.Datetime | pl.Time:
                return "datetime"
            case _:  # Fallback for any unmatched types
                # Use string representation and apply same logic as FastAPIUtils
                type_str = str(type).lower()
                if any(x in type_str for x in ["int", "float", "double", "decimal"]):
                    return "number"
                elif any(x in type_str for x in ["str", "string", "utf8"]):
                    return "string"
                elif any(x in type_str for x in ["bool", "boolean"]):
                    return "boolean"
                elif any(x in type_str for x in ["date", "time", "datetime"]):
                    return "datetime"
                elif "list" in type_str:
                    return "array"
                else:
                    return "string"  # Default fallback

    return {k: python_type_to_str(v) for k, v in schema.items()}


class Node:
    """
    A wrapper class for DataFrames and DocDataFrames
    that tracks parent-child relationships and supports lazy evaluation.

    The Node class supports multiple data types:
    - polars: DataFrame, LazyFrame
    - docframe: DocDataFrame
    - Any other data type with compatible methods

    Attributes:
        id: Unique identifier for the node
        name: Human-readable name for the node
        data: The underlying data (pl.DataFrame, pl.LazyFrame, or DocDataFrame)
        parents: List of parent nodes
        children: List of child nodes
        workspace: The containing workspace (never None)
        operation: Description of the operation that created this node
    """

    def __init__(
        self,
        data: SupportedDataTypes,
        name: str | None = None,
        workspace: "Workspace" | None = None,
        parents: list["Node"] | None = None,
        operation: str | None = None,
    ):
        """
        Initialize a new Node.

        Args:
            data: The underlying data to wrap
            name: Optional name for the node
            workspace: The containing workspace (creates new one if None)
            parents: List of parent nodes
            operation: Description of the operation that created this node
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"node_{self.id[:8]}"

        # Validate data type without relying on Union in isinstance (not supported)
        if not isinstance(
            data, (pl.DataFrame, pl.LazyFrame, DocDataFrame, DocLazyFrame)
        ):
            raise TypeError(
                f"Unsupported data type: {type(data).__name__}. "
                "Node only supports pl.DataFrame, pl.LazyFrame, DocDataFrame, DocLazyFrame."
            )

        self.data = data
        self.parents = parents or []
        self.children: List["Node"] = []

        # Ensure workspace is never None
        if workspace is None:
            from .workspace import Workspace  # Import here to avoid circular import

            workspace = Workspace(name=f"workspace_for_{self.name}")
        self.workspace: "Workspace" = workspace
        self.operation = operation

        # Add to workspace after fully initialized
        if self.id not in self.workspace.nodes:
            self.workspace.add_node(self)

        # Add this node as a child to all parents
        for parent in self.parents:
            parent.children.append(self)

    @property
    def is_lazy(self) -> bool:
        """Check if the node is in lazy state."""
        # Check based on the type of underlying data
        if isinstance(self.data, (pl.LazyFrame, DocLazyFrame)):
            return True
        elif isinstance(self.data, (pl.DataFrame, DocDataFrame)):
            return False
        else:
            raise TypeError(f"Unsupported data type: {type(self.data).__name__}")

    @property
    def document_column(self) -> Optional[str]:
        """
        Get the document column name if this is a DocDataFrame or DocLazyFrame.

        Returns:
            The document column name or None if not applicable.
        """
        if isinstance(self.data, (DocDataFrame, DocLazyFrame)):
            return self.data.document_column
        return None

    def collect(self) -> "Node":
        """
        Materialize a lazy node by calling collect() on LazyFrames.
        Creates a new node with the collected data to preserve the lazy/eager distinction.
        """
        if (
            self.is_lazy
            and hasattr(self.data, "collect")
            and callable(self.data.collect)
        ):
            try:
                collected_data = self.data.collect()
                # Create a new node with the collected data
                new_node = Node(
                    data=collected_data,
                    name=f"collect_{self.name}",
                    workspace=self.workspace,
                    parents=[self],
                    operation=f"collect({self.name})",
                )
                self.workspace.add_node(new_node)
                return new_node
            except (AttributeError, TypeError):
                # Handle case where collect is not available or not callable
                return self
        else:
            # If not lazy, return self
            return self

    def json_schema(self) -> Dict[str, str]:
        """
        Get the JSON schema of the node's data.

        Returns:
            Dictionary representing the schema in JSON format.
        """
        try:
            if self.is_lazy:
                schema = self.data.collect_schema()
            else:
                schema = self.data.schema
            return schema_to_json(schema)
        except Exception:
            # Return empty dict if schema extraction fails
            return {}

    def materialize(self) -> "Node":
        """
        Materialize a lazy node by calling collect() on LazyFrames in-place.
        This modifies the current node's data from lazy to eager.
        Returns self for method chaining.
        """
        if (
            self.is_lazy
            and hasattr(self.data, "collect")
            and callable(self.data.collect)
        ):
            try:
                self.data = self.data.collect()
            except (AttributeError, TypeError):
                # Handle case where collect is not available or not callable
                pass
        return self

    def join(self, other: "Node", **kwargs) -> "Node":
        """
        Join this node with another node using delegation to underlying data.

        If one node is lazy and the other is non-lazy, convert the lazy node to
        non-lazy (eager) before joining.

        Args:
            other: The other node to join with
            **kwargs: Additional arguments passed to the join operation

        Returns:
            A new Node containing the joined data
        """
        # Check if we have mixed lazy/non-lazy types and handle accordingly
        self_is_lazy = self.is_lazy
        other_is_lazy = other.is_lazy

        # If one is lazy and the other is not, convert lazy to non-lazy
        if self_is_lazy and not other_is_lazy:
            # Convert self (lazy) to non-lazy
            if isinstance(self.data, DocLazyFrame):
                self_data = self.data.collect()  # DocLazyFrame -> DocDataFrame
            elif isinstance(self.data, pl.LazyFrame):
                self_data = self.data.collect()  # LazyFrame -> DataFrame
            else:
                self_data = self.data
            other_data = other.data
        elif not self_is_lazy and other_is_lazy:
            # Convert other (lazy) to non-lazy
            self_data = self.data
            if isinstance(other.data, DocLazyFrame):
                other_data = other.data.collect()  # DocLazyFrame -> DocDataFrame
            elif isinstance(other.data, pl.LazyFrame):
                other_data = other.data.collect()  # LazyFrame -> DataFrame
            else:
                other_data = other.data
        else:
            # Both are same type (both lazy or both non-lazy) - use as is
            # For mixed types involving DocLazyFrame, extract underlying LazyFrame
            if isinstance(self.data, pl.LazyFrame) and isinstance(
                other.data, DocLazyFrame
            ):
                self_data = self.data
                other_data = other.data._df  # Extract underlying LazyFrame
            elif isinstance(self.data, DocLazyFrame) and isinstance(
                other.data, pl.LazyFrame
            ):
                self_data = self.data._df  # Extract underlying LazyFrame
                other_data = other.data
            elif isinstance(self.data, DocLazyFrame) and isinstance(
                other.data, DocLazyFrame
            ):
                self_data = self.data._df  # Extract underlying LazyFrame
                other_data = other.data._df  # Extract underlying LazyFrame
            # For mixed types involving DocDataFrame, extract underlying DataFrame
            elif isinstance(self.data, pl.DataFrame) and isinstance(
                other.data, DocDataFrame
            ):
                self_data = self.data
                other_data = other.data._df  # Extract underlying DataFrame
            elif isinstance(self.data, DocDataFrame) and isinstance(
                other.data, pl.DataFrame
            ):
                self_data = self.data._df  # Extract underlying DataFrame
                other_data = other.data
            elif isinstance(self.data, DocDataFrame) and isinstance(
                other.data, DocDataFrame
            ):
                self_data = self.data._df  # Extract underlying DataFrame
                other_data = other.data._df  # Extract underlying DataFrame
            else:
                self_data = self.data
                other_data = other.data

        # Perform the join using the processed data
        if hasattr(self_data, "join"):
            joined_data = self_data.join(other_data, **kwargs)
        else:
            raise TypeError(
                f"Join operation not supported for data types: {type(self_data)}, {type(other_data)}"
            )

        new_node = Node(
            data=joined_data,
            name=f"join_{self.name}_{other.name}",
            workspace=self.workspace,
            parents=[self, other],
            operation=f"join({self.name}, {other.name})",
        )

        # Add the new node to the workspace
        self.workspace.add_node(new_node)

        return new_node

    def filter(self, condition, **kwargs) -> "Node":
        """
        Filter the data based on a condition using delegation to underlying data.

        DocDataFrame and DocLazyFrame have delegation to polars methods,
        so we can call filter directly on them.

        Args:
            condition: Filter condition (function, boolean mask, or query string)
            **kwargs: Additional arguments passed to the filter operation

        Returns:
            A new Node containing the filtered data
        """
        # Use direct delegation - DocDataFrame/DocLazyFrame handle polars methods automatically
        if hasattr(self.data, "filter"):
            filtered_data = self.data.filter(condition, **kwargs)
        else:
            raise TypeError(f"Filtering not supported for {type(self.data)}")

        new_node = Node(
            data=filtered_data,
            name=f"filter_{self.name}",
            workspace=self.workspace,
            parents=[self],
            operation=f"filter({self.name})",
        )

        # Add the new node to the workspace
        self.workspace.add_node(new_node)

        return new_node

    def slice(self, *args, **kwargs) -> "Node":
        """
        Slice the data using delegation to underlying data.

        DocDataFrame and DocLazyFrame have delegation to polars methods,
        so we can call slice directly on them.

        Args:
            *args: Slice arguments (offset, length, etc.) - passed to underlying data's slice method
                   Can also accept a Python slice object which will be converted to offset/length
            **kwargs: Additional keyword arguments

        Returns:
            A new Node containing the sliced data
        """
        # Handle Python slice object
        if len(args) == 1 and isinstance(args[0], slice):
            slice_obj = args[0]
            offset = slice_obj.start or 0
            if slice_obj.stop is not None:
                length = slice_obj.stop - offset
                args = (offset, length)
            else:
                args = (offset,)

        # Use direct delegation - DocDataFrame/DocLazyFrame handle polars methods automatically
        if hasattr(self.data, "slice"):
            sliced_data = self.data.slice(*args, **kwargs)
        else:
            raise TypeError(f"Slice operation not supported for {type(self.data)}")

        new_node = Node(
            data=sliced_data,
            name=f"slice_{self.name}",
            workspace=self.workspace,
            parents=[self],
            operation=f"slice({self.name})",
        )

        # Add the new node to the workspace
        self.workspace.add_node(new_node)

        return new_node

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying data object.
        This allows Node to act as a proxy for the wrapped data.

        Operations that return DataFrame, LazyFrame, or DocDataFrame will be wrapped as new Nodes.
        All other return types are returned as-is.
        """
        if name == "columns" and self.is_lazy:
            return self.data.collect_schema().names()
        if hasattr(self.data, name):
            attr = getattr(self.data, name)

            # If it's a method that would return a new DataFrame,
            # wrap the result in a new Node
            if callable(attr):

                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)

                    # Check if result should be wrapped in a new Node
                    # Only supported data types (DataFrame, LazyFrame, DocDataFrame, DocLazyFrame) become Nodes
                    if isinstance(
                        result,
                        (
                            pl.DataFrame,
                            pl.LazyFrame,
                            DocDataFrame,
                            DocLazyFrame,
                            pl.Series,
                        ),
                    ):
                        if isinstance(result, pl.Series):
                            result = result.to_frame()
                        new_node = Node(
                            data=result,
                            name=f"{name}_{self.name}",
                            workspace=self.workspace,
                            parents=[self],
                            operation=f"{name}({self.name})",
                        )
                        # Add the new node to the workspace
                        self.workspace.add_node(new_node)
                        return new_node
                    # Handle intermediate polars objects (GroupBy, Rolling, etc.)
                    elif (
                        hasattr(result, "__class__")
                        and "polars" in str(result.__class__.__module__)
                        and not isinstance(
                            result,
                            (
                                pl.DataFrame,
                                pl.LazyFrame,
                                DocDataFrame,
                                DocLazyFrame,
                                pl.Series,
                            ),
                        )
                    ):
                        # Create a proxy that wraps methods returning DataFrames/LazyFrames
                        return self._create_intermediate_proxy(result, name)
                    else:
                        # Return non-Node-compatible results as-is
                        return result

                return wrapper
            else:
                return attr
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def _create_intermediate_proxy(self, intermediate_obj: Any, operation_name: str):
        """
        Create a proxy for intermediate polars objects (like LazyGroupBy)
        that wraps their methods to return Nodes when appropriate.

        Only methods that return DataFrame, LazyFrame, or DocDataFrame will be wrapped as Nodes.
        All other return types are returned as-is.
        """

        class IntermediateProxy:
            def __init__(self, obj, parent_node, op_name):
                self._obj = obj
                self._parent_node = parent_node
                self._operation_name = op_name

            def __getattr__(self, name):
                attr = getattr(self._obj, name)
                if callable(attr):

                    def wrapper(*args, **kwargs):
                        result = attr(*args, **kwargs)
                        # Only wrap results that are supported Node data types
                        if isinstance(
                            result, (pl.DataFrame, pl.LazyFrame, DocDataFrame)
                        ):
                            new_node = Node(
                                data=result,
                                name=f"{self._operation_name}_{name}_{self._parent_node.name}",
                                workspace=self._parent_node.workspace,
                                parents=[self._parent_node],
                                operation=f"{self._operation_name}.{name}({self._parent_node.name})",
                            )
                            self._parent_node.workspace.add_node(new_node)
                            return new_node
                        else:
                            # Return non-Node-compatible results as-is
                            return result

                    return wrapper
                else:
                    return attr

        return IntermediateProxy(intermediate_obj, self, operation_name)

    def __repr__(self) -> str:
        """String representation of the Node."""
        data_type = type(self.data).__name__
        parent_count = len(self.parents)
        child_count = len(self.children)

        return (
            f"Node(id={self.id[:8]}, name='{self.name}', dtype={data_type}, "
            f"lazy={self.is_lazy}, parents={parent_count}, children={child_count})"
        )

    def info(self, json: bool = False) -> Dict[str, Any]:
        """
        Return detailed information about the node.

        Returns:
            Dictionary containing node information
        """
        dtype = type(self.data)
        info_dict = {
            "id": self.id,
            "name": self.name,
            "dtype": dtype if not json else f"{dtype.__module__}.{dtype.__name__}",
            "lazy": self.is_lazy,
            "operation": self.operation,
            "parent_ids": [p.id for p in self.parents],
            "child_ids": [c.id for c in self.children],
        }

        # Add data-specific information
        # Only add shape for materialized data (DataFrame, not LazyFrame)
        if isinstance(self.data, (pl.DataFrame, DocDataFrame)) and hasattr(
            self.data, "shape"
        ):
            info_dict["shape"] = self.data.shape
        elif isinstance(self.data, (pl.LazyFrame, DocLazyFrame)):
            # Use underlying LazyFrame for DocLazyFrame
            lf = (
                self.data.lazyframe
                if isinstance(self.data, DocLazyFrame)
                else self.data
            )
            height = lf.select(pl.count()).collect().item()
            width = len(lf.collect_schema().names())
            info_dict["shape"] = (height, width)

        # Handle schema extraction differently for LazyFrames to avoid performance warnings
        schema = None
        if self.is_lazy:
            # Use underlying LazyFrame for DocLazyFrame
            if isinstance(self.data, DocLazyFrame):
                schema = self.data.lazyframe.collect_schema()
            else:
                schema = self.data.collect_schema()
        else:
            schema = self.data.schema

        if schema is not None:
            info_dict["schema"] = schema if not json else schema_to_json(schema)
        else:
            info_dict["schema"] = {} if json else {}

        if isinstance(self.data, (DocDataFrame, DocLazyFrame)):
            info_dict["document_column"] = self.document_column

        return info_dict

    # FastAPI-friendly methods
    # API-specific methods have been moved to ldaca_web_app backend
    # to keep docworkspace general-purpose

    def serialize(self, format: str = "json") -> Dict[str, Any]:
        """
        Serialize the node's data using the specified format.

        Args:
            format: Serialization format ('json' supported, 'binary' reserved for future)

        Returns:
            Dictionary containing serialized data and metadata
        """
        if format != "json":
            raise ValueError(
                f"Unsupported format: {format}. Currently only 'json' is supported."
            )

        # Node metadata
        node_metadata = {
            "id": self.id,
            "name": self.name,
            "operation": self.operation,
            "data_type": type(self.data).__name__,
            "is_lazy": self.is_lazy,
        }

        # Serialize the underlying data using its own JSON method
        if isinstance(self.data, (DocDataFrame, DocLazyFrame)):
            # Use docframe serialization (JSON only)
            serialized_data = self.data.serialize(format="json")
            data_metadata = {"type": type(self.data).__name__}
        elif isinstance(self.data, (pl.DataFrame, pl.LazyFrame)):
            # Use polars dict representation for JSON compatibility
            if isinstance(self.data, pl.LazyFrame):
                # Collect LazyFrame to get data for JSON serialization
                df_data = self.data.collect().to_dict(as_series=False)
            else:
                df_data = self.data.to_dict(as_series=False)

            serialized_data = df_data
            data_metadata = {
                "type": "polars_DataFrame"
                if isinstance(self.data, pl.DataFrame)
                else "polars_LazyFrame"
            }
        else:
            # Fallback - convert to string representation
            serialized_data = str(self.data)
            data_metadata = {"type": "string_fallback"}

        return {
            "node_metadata": node_metadata,
            "data_metadata": data_metadata,
            "serialized_data": serialized_data,
        }

    @classmethod
    def deserialize(
        cls,
        serialized_node: Dict[str, Any],
        workspace: "Workspace",
        format: str = "json",
    ) -> "Node":
        """
        Deserialize a node from serialized data.

        Args:
            serialized_node: Dictionary containing serialized node data
            workspace: The workspace to attach the node to
            format: Serialization format ('json' supported, 'binary' reserved for future)

        Returns:
            Deserialized Node object
        """
        if format != "json":
            raise ValueError(
                f"Unsupported format: {format}. Currently only 'json' is supported."
            )

        node_metadata = serialized_node["node_metadata"]
        data_metadata = serialized_node["data_metadata"]
        serialized_data = serialized_node["serialized_data"]

        # Deserialize the data based on its type
        if data_metadata["type"] == "DocDataFrame":
            # Use DocDataFrame's JSON deserialization
            data = DocDataFrame.deserialize(serialized_data, format="json")
        elif data_metadata["type"] == "DocLazyFrame":
            # Use DocLazyFrame's JSON deserialization
            data = DocLazyFrame.deserialize(serialized_data, format="json")
        elif data_metadata["type"] in ["polars_DataFrame", "polars_LazyFrame"]:
            # Reconstruct polars DataFrame/LazyFrame from dict
            df = pl.DataFrame(serialized_data)
            if data_metadata["type"] == "polars_LazyFrame":
                data = df.lazy()
            else:
                data = df
        elif data_metadata["type"] == "string_fallback":
            # This is a fallback case - create a simple dataframe with the string
            data = pl.DataFrame({"data": [serialized_data]})
        else:
            raise ValueError(f"Unknown data type: {data_metadata['type']}")

        # Create the node manually without triggering workspace registration
        node = cls.__new__(cls)
        node.id = node_metadata["id"]
        node.name = node_metadata["name"]
        node.data = data
        node.parents = []
        node.children = []
        node.workspace = workspace
        node.operation = node_metadata["operation"]

        # Add to workspace manually to avoid duplicates
        workspace.nodes[node.id] = node

        return node
