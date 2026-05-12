# Releasing to PyPI

This project is configured for PyPI releases under the package name
`gym-invmgmt`.

## One-time PyPI setup

Use PyPI Trusted Publishing so release jobs do not need long-lived API tokens.

1. Sign in to PyPI.
2. Open `https://pypi.org/manage/account/publishing/`.
3. Add a pending publisher with:
   - PyPI project name: `gym-invmgmt`
   - Owner: `r2barati`
   - Repository: `gym-invmgmt`
   - Workflow filename: `publish.yml`
   - GitHub environment: `pypi`
4. In the GitHub repository settings, create an environment named `pypi`.
   Requiring manual approval for this environment is recommended.

## Release checklist

1. Update `version` in `pyproject.toml`.
2. Update `__version__` in `gym_invmgmt/__init__.py`.
3. Move the relevant entries in `CHANGELOG.md` from `Unreleased` to the new
   version section.
4. Run the local release preflight:

   ```bash
   python -m pip install -e ".[dev,release]"
   pytest tests/ -q
   python -m build
   python -m twine check dist/*
   ```

5. Commit the release changes.
6. Create and push a version tag:

   ```bash
   git tag v0.2.0
   git push origin main --tags
   ```

The `.github/workflows/publish.yml` workflow builds the source distribution and
wheel for every manual run and publishes to PyPI only for tag refs.

## Manual upload fallback

Trusted Publishing is preferred. If you must upload manually, create a PyPI API
token scoped to the `gym-invmgmt` project and run:

```bash
python -m pip install -e ".[release]"
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Use `__token__` as the username and the PyPI API token as the password.
