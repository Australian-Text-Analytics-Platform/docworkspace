# Publishing docworkspace

This repository publishes `docworkspace` to PyPI with GitHub Actions trusted
publishing.

The release process follows the same split as `polars-text`:

- `.github/workflows/ci.yml` runs build-and-test validation on branch pushes,
  pull requests, and manual dispatches
- `.github/workflows/release.yml` builds the package once and publishes only on
  explicit version tags

`docworkspace` is a pure Python package, so the release workflow builds one
wheel and one source distribution with `uv build` instead of using
platform-specific wheel jobs.

## Release policy

- Pull requests: build and validate distributions only, no publish
- Tags matching `v*`: publish to PyPI
- Manual `workflow_dispatch` with `publish_target = testpypi`: publish the
  selected ref to TestPyPI

There is no automatic publish path from plain branch pushes.

## One-time setup

Before the first release, configure the package index side correctly.

### 1. Create the project on PyPI and optionally TestPyPI

Reserve the `docworkspace` project name.

### 2. Configure trusted publishing

Add a pending trusted publisher on PyPI with these values:

- Owner: `Australian-Text-Analytics-Platform`
- Repository: `docworkspace`
- Workflow file name: `release.yml`
- Environment: leave blank with the current workflow

If you want manual TestPyPI dry runs, repeat the same setup on TestPyPI.

### 3. Verify repository permissions

The publish jobs only need:

- `contents: read`
- `id-token: write`

## First release checklist

Before publishing `v0.1.0`:

1. Ensure `pyproject.toml` says `0.1.0`
2. Make sure `.github/workflows/release.yml` is already on the default branch
3. Add the pending trusted publisher on PyPI
4. Optionally add the same trusted publisher on TestPyPI
5. Verify the package builds locally:

```bash
uv sync --group dev
uv run pytest -q
uv build
```

## Manual TestPyPI dry run

Use this when you want to verify publishing without making a public release.

1. Open the `Package And Release` workflow in GitHub Actions
2. Choose `Run workflow`
3. Select the ref you want to test
4. Set `publish_target` to `testpypi`
5. Run the workflow

## Official release procedure

For the first stable release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The tag triggers the release workflow, which uploads the wheel and source
distribution to PyPI.

Verify installation after the publish completes:

```bash
python -m pip install docworkspace==0.1.0
```

## Hotfix release procedure

For a hotfix after `0.1.0`, bump `pyproject.toml` to the next patch version,
commit it, and publish the new tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

## RC / prerelease procedure

For a public prerelease, use a PEP 440 version in `pyproject.toml`, then tag
the matching Git ref:

```bash
git tag v0.2.0rc1
git push origin v0.2.0rc1
```

PyPI will treat the uploaded package as a prerelease automatically.
