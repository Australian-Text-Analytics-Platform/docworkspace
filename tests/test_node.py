"""Tests for the Node class."""

from typing import cast

import polars as pl
import pytest

from docworkspace import Node, TokenizationMeta, Workspace


class TestNode:
    """Test cases for the Node class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample polars DataFrame."""
        return pl.DataFrame({"text": ["Hello", "World", "Test"], "value": [1, 2, 3]})

    def test_node_creation_with_workspace(self, sample_df):
        """Test creating a Node with explicit workspace."""
        workspace = Workspace("test_workspace")
        node = Node(sample_df.lazy(), "test_node", workspace)

        assert node.name == "test_node"
        assert isinstance(node.data, pl.LazyFrame)
        assert len(node.parents) == 0
        assert len(node.children) == 0
        assert node.workspace == workspace

    def test_node_creation_with_string_parent_ids_without_workspace(self, sample_df):
        """Test creating an unattached Node with unresolved parent ids."""
        node = Node(
            sample_df.lazy(),
            "child_node",
            workspace=None,
            parents=["parent-123"],
        )

        assert node.workspace is None
        assert node.parents == ["parent-123"]
        assert node.children == []

    def test_node_filter(self, sample_df):
        """Test filtering a Node."""
        workspace = Workspace("test_workspace")
        node = Node(sample_df.lazy(), "test_node", workspace)

        # Filter using polars syntax
        filtered = node.filter(pl.col("value") > 1)

        assert len(filtered.parents) == 1
        assert filtered.parents[0] == node
        assert len(node.children) == 1
        assert node.children[0] == filtered
        assert filtered.workspace == workspace
        assert filtered.id in workspace.nodes

    def test_node_select_creates_child_with_selected_schema(self):
        """Selecting columns should create a child node in the same workspace."""
        workspace = Workspace("select_test")
        node = Node(
            pl.DataFrame(
                {"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10, 20, 30]}
            ).lazy(),
            "original",
            workspace,
        )

        selected = node.select(["id", "name"])

        assert selected is not node
        assert selected.workspace == workspace
        assert selected.parents == [node]
        assert selected in node.children
        assert selected.data.collect_schema().names() == ["id", "name"]

    def test_node_slice(self, sample_df):
        """Test slicing a Node."""
        node = Node(sample_df.lazy(), "test_node")
        sliced = node.slice(0, 2)

        assert len(sliced.parents) == 1
        assert sliced.parents[0] == node
        assert cast(pl.DataFrame, sliced.data.collect()).height == 2

    def test_node_drop_creates_child_and_drops_column(self, sample_df):
        """Dropping columns returns a child node with updated schema."""
        workspace = Workspace("test_workspace")
        node = Node(sample_df.lazy(), "test_node", workspace)

        dropped = node.drop("value")

        assert dropped is not node
        assert len(dropped.parents) == 1
        assert dropped.parents[0] == node
        assert dropped in node.children
        assert "value" not in dropped.data.collect_schema().names()

    def test_node_drop_clears_document_when_document_column_removed(self):
        """Dropping the document column clears document metadata on child."""
        workspace = Workspace("test_workspace")
        node = Node(
            pl.DataFrame({"text": ["a", "b"], "value": [1, 2]}).lazy(),
            "test_node",
            workspace,
        )
        node.document = "text"

        dropped = node.drop("text")

        assert dropped.document is None

    def test_node_drop_preserves_document_when_other_column_removed(self):
        """Dropping a non-document column preserves child document metadata."""
        workspace = Workspace("test_workspace")
        node = Node(
            pl.DataFrame({"text": ["a", "b"], "value": [1, 2]}).lazy(),
            "test_node",
            workspace,
        )
        node.document = "text"

        dropped = node.drop("value")

        assert dropped.document == "text"

    def test_node_rename_is_in_place_and_renames_column(self, sample_df):
        """Renaming columns mutates the node in-place and updates schema."""
        workspace = Workspace("test_workspace")
        node = Node(sample_df.lazy(), "test_node", workspace)

        renamed = node.rename({"value": "score"})

        columns = renamed.data.collect_schema().names()
        assert "score" in columns
        assert "value" not in columns
        assert renamed is node

    def test_node_rename_updates_document_metadata(self):
        """Renaming the document column updates node document metadata."""
        workspace = Workspace("test_workspace")
        node = Node(
            pl.DataFrame({"text": ["a", "b"], "value": [1, 2]}).lazy(),
            "test_node",
            workspace,
        )
        node.document = "text"

        renamed = node.rename({"text": "content"})

        assert renamed.document == "content"
        assert renamed is node

    def test_node_rename_preserves_document_when_other_column_renamed(self):
        """Renaming a non-document column preserves node document metadata."""
        workspace = Workspace("test_workspace")
        node = Node(
            pl.DataFrame({"text": ["a", "b"], "value": [1, 2]}).lazy(),
            "test_node",
            workspace,
        )
        node.document = "text"

        renamed = node.rename({"value": "score"})

        assert renamed.document == "text"
        assert renamed is node

    def test_node_join(self):
        """Test joining two Nodes."""
        df1 = pl.DataFrame({"key": ["A", "B"], "value1": [1, 2]})
        df2 = pl.DataFrame({"key": ["A", "B"], "value2": [3, 4]})

        workspace = Workspace("test_workspace")
        node1 = Node(df1.lazy(), "node1", workspace)
        node2 = Node(df2.lazy(), "node2", workspace)

        # Polars uses join instead of merge
        merged = node1.join(node2, on="key")

        assert len(merged.parents) == 2
        assert node1 in merged.parents
        assert node2 in merged.parents
        assert merged.data.collect_schema().len() == 3  # key, value1, value2

    def test_node_attribute_delegation(self, sample_df):
        """Test that Node delegates attributes to the underlying data."""
        node = Node(sample_df.lazy(), "test_node")

        # Test property access
        assert node.shape == (3, 2)
        assert list(node.columns) == list(sample_df.columns)

        # Test method call that returns a new DataFrame
        head_node = node.head(2)
        assert isinstance(head_node, Node)
        assert head_node.data.collect().height == 2
        assert head_node.parents[0] == node

    def test_node_delegation_returns_raw_non_lazyframe_results(self, sample_df):
        """Delegated methods that do not return LazyFrame stay unwrapped."""
        node = Node(sample_df.lazy(), "test_node")

        schema = node.collect_schema()

        assert isinstance(schema, pl.Schema)
        assert schema.names() == list(sample_df.columns)

    def test_node_select_preserves_tokenization_metadata_by_source_column(self):
        """Selecting a tokenized source column keeps its tokenization spec."""
        meta: TokenizationMeta = {
            "column_name": "tokenization.text.lindera:jieba",
            "model": "lindera:jieba",
            "language": "zh",
            "params": {"lowercase": False, "remove_punct": True},
        }
        node = Node(
            pl.DataFrame({"text": ["左"], "other": [1]}).lazy(),
            "source",
            tokenization={"text": meta},
        )

        selected = node.select("text")

        assert selected.tokenization == {"text": meta}
        assert selected.find_tokenization_column("text") == "tokenization.text.lindera:jieba"
        assert "tokenization.text.lindera:jieba" not in selected.data.collect_schema().names()

    def test_node_select_drops_tokenization_metadata_without_source_column(self):
        """Selecting away a source column invalidates only that tokenization spec."""
        text_meta: TokenizationMeta = {
            "column_name": "tokenization.text.lindera:jieba",
            "model": "lindera:jieba",
            "language": "zh",
            "params": {"lowercase": True, "remove_punct": True},
        }
        other_meta: TokenizationMeta = {
            "column_name": "tokenization.other.huggingface:bert-base-uncased",
            "model": "huggingface:bert-base-uncased",
            "language": "en",
            "params": {"lowercase": True, "remove_punct": True},
        }
        node = Node(
            pl.DataFrame({"text": ["左"], "other": ["right"]}).lazy(),
            "source",
            tokenization={"text": text_meta, "other": other_meta},
        )

        selected = node.select("other")

        assert selected.tokenization == {"other": other_meta}

    def test_node_drop_cascades_tokenization_columns(self):
        """Dropping a tokenized source removes metadata and hydrated column if present."""
        meta: TokenizationMeta = {
            "column_name": "tokenization.text.lindera:jieba",
            "model": "lindera:jieba",
            "language": "zh",
            "params": {"lowercase": True, "remove_punct": True},
        }
        node = Node(
            pl.DataFrame(
                {
                    "text": ["左"],
                    "other": [1],
                    "tokenization.text.lindera:jieba": [
                        [{"token": "左", "start": 0, "end": 1}]
                    ],
                }
            ).lazy(),
            "source",
            tokenization={"text": meta},
        )

        survivor = node.drop("other")
        assert survivor.tokenization == {"text": meta}

        cascaded = node.drop("text")
        assert cascaded.tokenization == {}
        assert "tokenization.text.lindera:jieba" not in cascaded.data.collect_schema().names()

    def test_node_rename_cascades_tokenization_columns(self):
        """Renaming a tokenized source removes stale tokenization metadata."""
        meta: TokenizationMeta = {
            "column_name": "tokenization.text.lindera:jieba",
            "model": "lindera:jieba",
            "language": "zh",
            "params": {"lowercase": True, "remove_punct": True},
        }
        node = Node(
            pl.DataFrame(
                {
                    "text": ["左"],
                    "tokenization.text.lindera:jieba": [
                        [{"token": "左", "start": 0, "end": 1}]
                    ],
                }
            ).lazy(),
            "source",
            tokenization={"text": meta},
        )

        node.rename({"text": "body"})

        after_names = node.data.collect_schema().names()
        assert node.tokenization == {}
        assert "body" in after_names
        assert "tokenization.text.lindera:jieba" not in after_names

    def test_node_join_rejects_conflicting_tokenization_metadata(self):
        """Joining nodes must not silently overwrite tokenization specs."""
        left_meta: TokenizationMeta = {
            "column_name": "tokenization.text.lindera:jieba",
            "model": "lindera:jieba",
            "language": "zh",
            "params": {"lowercase": True, "remove_punct": True},
        }
        right_meta: TokenizationMeta = {
            "column_name": "tokenization.text.huggingface:bert-base-uncased",
            "model": "huggingface:bert-base-uncased",
            "language": "en",
            "params": {"lowercase": True, "remove_punct": True},
        }
        left = Node(
            pl.DataFrame({"key": [1], "text": ["左"]}).lazy(),
            "left",
            tokenization={"text": left_meta},
        )
        right = Node(
            pl.DataFrame({"key": [1], "text": ["right"]}).lazy(),
            "right",
            tokenization={"text": right_meta},
        )

        with pytest.raises(ValueError, match="Conflicting tokenization metadata"):
            left.join(right, on="key")

    def test_node_join_merges_distinct_tokenization_metadata(self):
        """Joining distinct tokenized source columns preserves both specs."""
        text_meta: TokenizationMeta = {
            "column_name": "tokenization.text.lindera:jieba",
            "model": "lindera:jieba",
            "language": "zh",
            "params": {"lowercase": True, "remove_punct": True},
        }
        other_meta: TokenizationMeta = {
            "column_name": "tokenization.other.huggingface:bert-base-uncased",
            "model": "huggingface:bert-base-uncased",
            "language": "en",
            "params": {"lowercase": True, "remove_punct": True},
        }
        left = Node(
            pl.DataFrame({"key": [1], "text": ["左"]}).lazy(),
            "left",
            tokenization={"text": text_meta},
        )
        right = Node(
            pl.DataFrame({"key": [1], "other": ["right"]}).lazy(),
            "right",
            tokenization={"other": other_meta},
        )

        joined = left.join(right, on="key")

        assert joined.tokenization == {"text": text_meta, "other": other_meta}

    def test_node_info(self, sample_df):
        """Test node info method returns JSON-safe dict."""
        workspace = Workspace("test_workspace")
        node = Node(sample_df.lazy(), "test_node", workspace, operation="load")

        info = node.info()

        assert info["name"] == "test_node"
        assert info["operation"] == "load"
        assert info["shape"] == (3, 2)
        assert info["document"] is None
        # Schema should be a dict of column name -> string type
        assert isinstance(info["schema"], dict)
        assert len(info["schema"]) == 2
        assert all(isinstance(v, str) for v in info["schema"].values())
        # Columns should be a list of column names
        assert info["columns"] == ["text", "value"]

    def test_node_data_setter_creates_undo_checkpoint(self, sample_df):
        """Assigning node.data should push previous plan onto undo stack."""
        node = Node(sample_df.lazy(), "test_node")

        node.data = node.data.with_columns(pl.lit("x").alias("new_col"))

        assert node.can_undo is True
        assert node.can_redo is False
        assert "new_col" in node.data.collect_schema().names()

    def test_node_undo_and_redo_round_trip(self, sample_df):
        """Undo should restore prior plan and redo should reapply it."""
        node = Node(sample_df.lazy(), "test_node")

        original_columns = list(node.data.collect_schema().names())
        node.data = node.data.with_columns(pl.lit(99).alias("new_col"))
        changed_columns = list(node.data.collect_schema().names())

        assert changed_columns != original_columns

        node.undo()
        assert list(node.data.collect_schema().names()) == original_columns
        assert node.can_redo is True

        node.redo()
        assert list(node.data.collect_schema().names()) == changed_columns

    def test_new_assignment_clears_redo_stack(self, sample_df):
        """A fresh assignment after undo should clear redo history."""
        node = Node(sample_df.lazy(), "test_node")

        node.data = node.data.with_columns(pl.lit(1).alias("c1"))
        node.data = node.data.with_columns(pl.lit(2).alias("c2"))
        node.undo()
        assert node.can_redo is True

        node.data = node.data.with_columns(pl.lit(3).alias("c3"))
        assert node.can_redo is False

    def test_undo_raises_when_history_empty(self, sample_df):
        """Undo should fail clearly when there is no undo history."""
        node = Node(sample_df.lazy(), "test_node")

        with pytest.raises(ValueError, match="Nothing to undo"):
            node.undo()

    def test_redo_raises_when_history_empty(self, sample_df):
        """Redo should fail clearly when there is no redo history."""
        node = Node(sample_df.lazy(), "test_node")

        with pytest.raises(ValueError, match="Nothing to redo"):
            node.redo()

    def test_node_info_includes_undo_redo_flags(self, sample_df):
        """Node info payload should expose can_undo/can_redo for API/UI use."""
        node = Node(sample_df.lazy(), "test_node")

        initial_info = node.info()
        assert initial_info["can_undo"] is False
        assert initial_info["can_redo"] is False

        node.data = node.data.with_columns(pl.lit(1).alias("new_col"))
        updated_info = node.info()
        assert updated_info["can_undo"] is True
        assert updated_info["can_redo"] is False


class TestNodeRelationships:
    """Test parent-child relationships between nodes."""

    @pytest.fixture
    def workspace(self):
        """Create a test workspace."""
        return Workspace("test_workspace")

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "B", "C"],
                "value": [10, 20, 30, 40, 50],
            }
        )

    def test_multiple_children(self, workspace, sample_df):
        """Test that a node can have multiple children."""
        parent = Node(sample_df.lazy(), "parent", workspace)

        child1 = parent.filter(pl.col("category") == "A")
        child2 = parent.filter(pl.col("category") == "B")
        child3 = parent.slice(0, 3)

        assert len(parent.children) == 3
        assert child1 in parent.children
        assert child2 in parent.children
        assert child3 in parent.children

    def test_children_property_reflects_parent_relationship_changes(
        self, workspace, sample_df
    ):
        """Children should be derived from workspace parent links, not cached state."""
        parent = Node(sample_df.lazy(), "parent", workspace)
        child = parent.filter(pl.col("category") == "A")

        assert child in parent.children

        child.parents = []

        assert parent.children == []

    def test_add_node_resolves_string_parent_ids(self, workspace, sample_df):
        """Attaching an unattached node should resolve matching parent ids."""
        parent = Node(sample_df.lazy(), "parent", workspace)
        child = Node(
            sample_df.lazy(),
            "child",
            workspace=None,
            parents=[parent.id],
        )

        workspace.add_node(child)

        assert child.workspace == workspace
        assert child.parents == [parent]
        assert child in parent.children


def test_node_shape_does_not_materialise_list_columns():
    """Phase 2.8: Node.shape must compute height without scanning list columns.

    A future change that pushes Node.shape towards full collect() (e.g. via
    .height instead of .select(pl.len())) would break tokenised nodes by
    forcing the List[Struct] column to be materialised. Bound it loosely at
    100ms on a 50k-row tokenised-style frame.
    """
    import time

    N = 50_000
    tokens_per_doc = 30
    tokens_struct = [
        {"token": f"t{i}", "start": i * 5, "end": i * 5 + 4}
        for i in range(tokens_per_doc)
    ]
    df = pl.DataFrame(
        {
            "text": [f"doc {i} " * 5 for i in range(N)],
            "TOKENS_tokens": [tokens_struct] * N,
        }
    )
    node = Node(data=df.lazy(), name="bench")

    start = time.perf_counter()
    shape = node.shape
    elapsed = time.perf_counter() - start

    assert shape == (N, 2)
    # Generous bound — typical observed time on dev hardware is < 1ms.
    assert elapsed < 0.1, (
        f"Node.shape took {elapsed * 1000:.1f}ms; suspect materialisation regression"
    )
