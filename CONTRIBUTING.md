# Contributing to gym-invmgmt

Thank you for your interest in contributing! This project follows the [Gymnasium](https://gymnasium.farama.org/) ecosystem conventions.

## Development Setup

```bash
git clone https://github.com/r2barati/gym-invmgmt.git
cd gym-invmgmt
pip install -e ".[dev]"
pre-commit install
```

## Making Changes

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality in `tests/test_env.py`.
3. **Run the test suite** before submitting:
   ```bash
   python -m pytest tests/test_env.py -v
   ```
4. **Run linters** (enforced in CI):
   ```bash
   ruff check gym_invmgmt/ tests/
   black --check gym_invmgmt/ tests/
   ```
5. **Submit a pull request** against the `main` branch.

## Code Quality

This project uses:
- **[Ruff](https://docs.astral.sh/ruff/)** for linting and import sorting
- **[Black](https://black.readthedocs.io/)** for formatting (line length: 120)
- **[Mypy](https://mypy.readthedocs.io/)** for type checking
- **[pre-commit](https://pre-commit.com/)** hooks for automated checks

All checks run automatically on PRs via GitHub Actions CI (Python 3.8–3.11).

## What to Contribute

- **New network topologies** — Add YAML configs to `gym_invmgmt/topologies/`
- **Demand patterns** — Extend `DemandEngine` with new composable effects
- **Wrappers** — Add new wrappers to `gym_invmgmt/wrappers/`
- **Bug fixes** — File an issue first, then submit a PR
- **Documentation** — Improvements to `docs/` or README

## Guidelines

- Follow the existing code style (Black-formatted, type-hinted).
- Keep backwards compatibility — avoid breaking changes to `gymnasium.make()` IDs or the public Python API.
- All new environments must pass `gymnasium.utils.env_checker.check_env()`.
- Include docstrings for all public functions and classes.

## Reporting Bugs

Please open an [issue](https://github.com/r2barati/gym-invmgmt/issues) with:
- Python and Gymnasium versions
- Minimal reproduction code
- Expected vs. actual behavior
