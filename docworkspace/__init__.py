"""DocWorkspace

A library for managing DataFrames and DocDataFrames with parent-child relationships
and lazy evaluation capabilities.
"""

from .node import Node
from .workspace import Workspace

__version__ = "0.1.0"
__all__ = ["Node", "Workspace"]
