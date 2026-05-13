# Releasing to PyPI

This project is configured for PyPI releases under the package name
`gym-invmgmt`: https://pypi.org/project/gym-invmgmt/

The first PyPI release, `0.2.0`, was published from the `v0.2.0` tag via
GitHub Trusted Publishing.

## One-time PyPI setup

Use PyPI Trusted Publishing so release jobs do not need long-lived API tokens.
This setup is already in place for the `gym-invmgmt` project. Keep the details
below as the source of truth if the publisher ever needs to be recreated.

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
   git tag vX.Y.Z
   git push origin main --tags
   ```

The `.github/workflows/publish.yml` workflow builds the source distribution and
wheel for every manual run and publishes to PyPI only for tag refs.

7. After the workflow succeeds, verify the published package from a clean
   environment:

   ```bash
   python -m pip install --upgrade gym-invmgmt
   python - <<'PY'
   import gymnasium as gym
   import gym_invmgmt

   env = gym.make("GymInvMgmt/MultiEchelon-v0")
   obs, _ = env.reset(seed=42)
   print(gym_invmgmt.__version__, obs.shape, env.action_space.shape)
   PY
   ```

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
