# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] — 2026-07-12

### Fixed
- `__version__` was hardcoded and could drift from the packaged version; it is
  now derived from the installed metadata (single source of truth).

### Changed
- A configured OpenAI judge whose calls all fail (bad key, no network, quota)
  now raises `JudgeError` instead of silently falling back to lexical scores and
  gating on them — a broken key can no longer quietly pass or fail a build.

## [0.1.1] — 2026-07-12

### Fixed
- README used relative links to `LICENSE`, `CONTRIBUTING.md`, and `pyproject.toml`,
  which 404'd on the PyPI project page. Point them at absolute GitHub URLs so they
  resolve on both GitHub and PyPI.

## [0.1.0] — 2026-07-12

Initial release.

### Added
- `raggate init | run | gate` CLI.
- Golden-set runner that calls a user-provided `target.run(question) -> dict`.
- Five scorers: `faithfulness`, `answer_relevancy`, `citation_support`,
  `context_coverage`, `answer_correctness`. Each runs as a lexical heuristic
  with no API key, or as an LLM-as-judge rubric when `OPENAI_API_KEY` is set.
- Band-based `FAIL` / `WARN` / `PASS` gates; only KPI metrics can block, and
  only in LLM-judge mode (heuristic mode is informational).
- Pinned, recorded judge config (`model` / `runs` / `temperature`) with
  multi-run averaging to damp judge variance.
- `evals/gates.high-stakes.yaml` profile for regulated domains.
- GitHub Action, `py.typed`, and a test suite covering the scorers, gate bands,
  config merge, score parsing, and I/O normalization.

### Notes
- `context_coverage` and `citation_support` are deliberately named for the
  cheaper proxies this kit computes; they are **not** the RAGAS `context_recall`
  / NLI citation metrics. See the README metric table.

[Unreleased]: https://github.com/abhay23-AI/raggate/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/abhay23-AI/raggate/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/abhay23-AI/raggate/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/abhay23-AI/raggate/releases/tag/v0.1.0
