# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-04

### Changed

- Updated release metadata for the environment-only package.
- Corrected documented observation dimensions for the built-in divergent and serial environments.

## [0.1.0] — 2026-03-15

### Added

- **Core environment** (`CoreEnv`) — Gymnasium-compatible multi-echelon inventory management environment with configurable topology, composable demand engine, and endogenous goodwill dynamics.
- **Built-in topologies** — `network` (divergent multi-echelon) and `serial` (linear chain) scenarios.
- **Custom topologies** — YAML-based network definition with auto-validation (DAG check, attribute requirements, node role detection).
- **Demand engine** — Composable non-stationary demand effects (trend, seasonal, shock), external data series support, noise scaling, and endogenous goodwill feedback.
- **Wrappers** — `IntegerActionWrapper` for discrete orders, `EpisodeLoggerWrapper` for trajectory recording.
- **Visualization** — `plot_network()` for topology rendering, `render_rgb_array()` for live KPI dashboard.
- **Video recording** — `rgb_array` render mode compatible with `gymnasium.wrappers.RecordVideo`.
- **Gymnasium registration** — `GymInvMgmt/MultiEchelon-v0` and `GymInvMgmt/Serial-v0`.
- **Input validation** — YAML config error handling, external data series validation (type, shape, NaN/Inf checks).
- **Test suite** — 16 tests covering registration, episodes, YAML parsing, demand engine, wrappers, seeding, and `check_env` compliance.
- **CI pipeline** — GitHub Actions for Python 3.8–3.11 with Ruff, Black, and automated env checker.
- **Documentation** — MDP formulation, demand engine guide, external datasets guide, network topologies reference, comparison with prior work, getting started tutorial, and visual dynamics guides.
