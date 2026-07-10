# Changelog

All notable changes to `dcleaner` are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-07-10

### Added
- `import dcleaner` shim so both `import dcleaner` (then `dcleaner.Data`)
  and `from dclean import Data` work, removing the pip-name/import-name
  confusion.

## [0.1.0] - 2026-07-10

### Added
- Initial release of `dcleaner`: a fluent data-cleaning and visualization
  layer on top of pandas.
- `Data` fluent API: `load`, `head`/`tail`/`shape`/`cols`/`describe`,
  `dropna`/`fillna`/`dedupe`/`drop`/`keep`/`rename`/`lower_cols`/`astype`,
  `filter` (expression strings), `mutate`/`select`/`sort`,
  `groupby`/`agg`/`summarize`/`corr`, `plot` (line/bar/hist/scatter/box/pie)
  and `plot_corr` heatmap, `savefig`/`show`, `to_csv`/`to_df`.
- Auto-format loading for CSV / Excel / JSON / Parquet.
- Headless-safe matplotlib backend (plots save in scripts/CI/servers).
- MIT license, PyPI-ready `pyproject.toml`, 8-test suite, demo notebook,
  and GitHub Actions CI (Python 3.8–3.11).
