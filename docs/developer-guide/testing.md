# Testing And Release

## Test Coverage Shape

The test suite covers:

- node construction, LazyFrame enforcement, delegation, filter/slice/join/drop,
  rename, undo/redo, and tokenization metadata behavior;
- workspace add/remove, parent resolution, serialization round trips, and JSON
  summaries;
- workspace path rebasing after folder moves and renames;
- save-time garbage collection and preservation of dotfile cache artifacts;
- backwards-compatible package exports and workspace shim imports.

## Local Commands

From `docworkspace/`:

```bash
uv run pytest -q
uvx ty check
```

CI checks out `polars-text` and `polars-source-utils` beside `docworkspace`
before running `uv sync --frozen --group dev`, matching the relative paths in
`tool.uv.sources`. Do not use `uv sync --no-sources` for coordinated branch CI
unless the bumped dependency versions have already been published.

For distribution checks:

```bash
uv build --no-sources
```

## Release Notes

`PUBLISH.md` documents PyPI trusted publishing. `docworkspace` is a pure Python
package, so release artifacts are a wheel and source distribution. The package
depends on `polars-text` and `polars-source-utils` for plan source-path
inspection and rewriting.

Before release, make sure `pyproject.toml` carries the intended version and the
tests pass from the package directory. Publish `polars-source-utils` and
`polars-text` first so `uv build --no-sources` validates package metadata
against versions that are available from PyPI.
