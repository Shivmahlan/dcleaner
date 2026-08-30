# Changelog

All notable changes to `dcleaner` are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Recipes - replay a pipeline on another file.** `save_recipe("monthly.json")`
  writes the steps this `Data` was built from; `apply_recipe("monthly.json")`
  replays them onto a different file:

  ```python
  Data("jan.csv").clean().filter("units > 5").save_recipe("monthly.json")
  Data("feb.csv").apply_recipe("monthly.json")
  ```

  **Design.** `.steps()` records display strings - `"filter('units > 5')"` -
  which are lossy: `"drop(['a', 'b'])"` and `"sort('a')"` cannot be turned back
  into a call without writing a parser for dclean's own repr, and that parser
  would be wrong the first time a column name contains a bracket or a quote.
  So a structured record is kept **alongside** the display string rather than
  parsed back out of it. Every step is now a dict - `method`, `args`, `kwargs`,
  `display` - and `_derive()` takes the structured call from the transform that
  already knows its own arguments. `steps()` still returns the same list of
  strings it always did, and `log()` prints what it always printed.

  **Only whitelisted methods replay.** The whitelist is the transforms that are
  a pure function of the frame plus JSON-serializable arguments. `join()` and
  `concat()` are NOT on it - they need a second table that a recipe cannot
  carry - and neither is `groupby().agg()`, which is two calls; `save_recipe()`
  refuses up front, naming the step and (for `groupby`) the one-call
  `mean(col, by=...)` form that does record. The whitelist is checked again at
  replay time, because the file may have been edited in between; an unknown
  method is an error, never a silent skip.

  **A recipe is code.** `filter()` and `mutate()` expressions are stored
  verbatim and evaluated on replay, so a recipe file is as trusted as a Python
  script - the existing `__` guard still applies, but it is a guard rail, not a
  sandbox. Run recipes you wrote, not recipes you were sent.

- **A command line: `dcleaner`.** `pip install dcleaner` now puts a `dcleaner`
  command on your PATH for the jobs that never needed a Python session:
  `dcleaner report FILE [--no-examples] [--html OUT]`,
  `dcleaner clean FILE [-o OUT] [--nulls keep|drop|fill] [--no-dates] [-q]` and
  `dcleaner head FILE [-n N]`, plus `--version`. Each subcommand is a thin
  wrapper over `dclean.report` / `dclean.clean` / `Data`, so the terminal and
  the library cannot disagree. `-o` picks its writer from the extension
  (csv/xlsx/json/parquet). It behaves like a shell tool: exit 0 on success,
  exit 1 with ONE line on stderr for a missing file, an unreadable file type or
  an output extension it cannot write - a traceback would mean a bug in dclean,
  not a typo in a filename. Built on stdlib `argparse`; no new dependency.
- **`report(to="profile.html")` - a profile you can send to someone.** The same
  per-column dtypes, null counts, unique counts, duplicate count, numeric stats
  and warnings the terminal prints, written as ONE self-contained HTML file:
  styles inline, no CDN link, nothing fetched at open time, so it survives an
  email attachment, a ticket and a firewall. `examples=False` is honoured there
  too, and matters more - that file is the one somebody else ends up holding.
  Values and column names taken from your data are HTML-escaped. Default
  behaviour is untouched: with no `to=`, `report()` prints exactly what it
  printed before.
- **Export parity with loading.** `dclean` reads csv/csv.gz/xls/xlsx/json/parquet
  but could only write CSV. `to_excel(path)`, `to_json(path)` and
  `to_parquet(path)` join `to_csv(path)`, all with the same contract: write the
  file, return the **same** object so an export can sit mid-chain. `openpyxl`
  and `pyarrow` stay optional - they are imported inside the method, and the
  error names the exact `pip install` when one is missing. `to_json` defaults to
  `orient="records"` with ISO-8601 dates, so dates that `clean()` parsed read
  back as dates rather than epoch milliseconds.
- **`_repr_html_` on `Data`** - a bare `d` at the end of a notebook cell now
  renders a real table (header line with the shape, plus the first 10 rows)
  instead of the terse one-line repr. `repr()` and `str()` are unchanged, so
  the terminal behaves exactly as before.
- A **Run in Colab** badge on `examples/demo.ipynb`, so the tour runs with
  nothing installed locally.

### Fixed
- **Bold/underline escape codes no longer leak into piped output.** They were
  emitted unconditionally, so `dcleaner report x.csv > profile.txt`, a pager or
  a CI log got `\033[1m` wrapped around every heading. The codes are now
  resolved once at import and are empty unless stdout is a terminal;
  `NO_COLOR` turns them off on a terminal and `FORCE_COLOR` turns them back on
  through a pipe (`FORCE_COLOR=0` forces them off).

### Changed
- **Plots now appear where you are working, instead of only in a saved PNG.**
  `dclean` no longer pins matplotlib to the `Agg` backend at import. In a
  Jupyter/IPython notebook a chart renders **inline** the moment you call
  `.plot()` / `.plot_corr()` / `.nulls(plot=True)`; from a script or REPL,
  `.show()` opens it in a window (it was silently a no-op before, which left
  `savefig()` + opening the file as the only way to look at a chart). With no
  display available, matplotlib still falls back to a file-only backend, so
  `savefig()` keeps working in scripts, CI and servers - and `.show()` now says
  why it cannot show anything rather than doing nothing.
  An explicitly chosen backend (`matplotlib.use(...)` before importing
  `dclean`, or `MPLBACKEND`) is always respected.

### Added
- **`join()` - combine two tables, and find out what actually matched.**
  `d.join("regions.csv", on="city")` (also `how="inner"/"right"/"outer"`,
  multi-column keys, and `left_on=`/`right_on=` when the two sides spell the key
  differently). `other` can be a `Data`, a DataFrame or a path; `on` defaults to
  the shared columns. The report is the feature: how many rows on each side
  matched, the keys that did not (with a sample), how many of those *would*
  match after case/whitespace folding, rows dropped by the `how`, null keys, and
  a warning when a non-unique right key multiplied your rows. Keys whose types
  can never match are refused up front instead of returning an empty frame.
  `verbose=False` skips the report and the counting work with it.
- **`concat()` - stack tables on top of each other.** `Data.concat("data/*.csv")`
  (globs, paths, `Data`s and DataFrames, or a list of them), or `d.concat(other)`
  on an existing `Data`. `source_col="file"` records which source each row came
  from. Reports the column that is missing from one source and the column whose
  type changes between sources - the two things that quietly ruin a stacked
  dataset.
- `show=` on `plot()`, `plot_corr()` and `nulls(plot=True)`: `None` (default)
  picks the sensible thing for where you are running, `True` displays it now,
  `False` builds the figure without displaying it.
- `to_fig()` - hands back the matplotlib `Figure`, the plotting counterpart of
  `to_df()`, for anything `dclean` doesn't wrap.

## [0.2.0] - 2026-08-29

### Changed - BREAKING
- **Transforms no longer mutate in place.** Every transform (`dropna`, `filter`,
  `mutate`, `to_float`, `agg`, `clean`, ...) now returns a **new** `Data`; the
  object you called it on is left exactly as it was. Previously `d.dropna()`
  modified `d` itself, so `clean = raw.dropna()` silently changed `raw` too —
  the opposite of the pandas contract and a real source of wrong results.
  Chained pipelines are unaffected; statement-style calls must now bind the
  result (`d = d.to_float("price")`).
- Inspect (`head`, `report`, `nulls`, ...), plot and output methods still return
  the same object, since they change no data. The rule is documented in the
  README's "Transforms never mutate" table.

### Added
- **`clean()` — one-call auto-clean.** Normalizes column names, drops all-empty
  rows/columns, trims whitespace, converts `""`/`"n/a"`/`"null"` to real NaN,
  coerces numeric-looking text to numbers (currency symbols and thousands
  separators included: `"$1,234.50"` → `1234.5`), parses date-looking columns to
  datetimes, removes duplicates, and handles nulls via
  `nulls="keep"|"drop"|"fill"`. Prints a report of every action taken.
- **`report()` — one-call data profile.** Per-column dtype, null count and
  percentage, unique count and an example value, plus duplicate-row count,
  numeric summary stats and data-quality warnings (constant columns, mostly-null
  columns, numbers stored as text, ID-like columns).
- **Module-level helpers** so the whole job is a single call:
  `dclean.clean(src, to="out.csv")`, `dclean.report(src)`, `dclean.load(src)`.
  Also exported from the `dcleaner` shim.
- `fix_nulls([strategy], [subset])` — fills missing values without you choosing a
  statistic per column (`auto`/`mean`/`median`/`mode`/`zero`/`ffill`).
- `drop_outliers([cols], [method], [factor])` — IQR (default) or z-score.
- `log()` / `steps()` — replay the pipeline that produced the current frame, so
  the toolkit's automatic decisions stay auditable.
- **One-call aggregates**, so the common questions cost one call instead of two:
  `mean/sum/count/median/min/max([col], [by])`, plus `top(n, by)`,
  `bottom(n, by)` and `counts(col)` (frequency table). All share one `stat()`
  engine; `.groupby().agg()` still works and returns identical results.
- `plot()` infers `x` and `y` on a two-column frame — what a grouped aggregate
  leaves you with — so `.mean("price", by="city").plot("bar")` needs no axes.
- `Data.samples()` — list the datasets bundled with the package.
- `copy()` — an explicit independent copy.

### Fixed
- **The bundled sample dataset now actually ships.** `dclean/data/sample_sales.csv`
  was declared in packaging and promised in the README since 0.1.3 but was never
  committed, so the README's headline example raised `FileNotFoundError` after a
  fresh `pip install`. The dataset is deliberately messy (padded column names,
  `$1,234.50` prices, `"n/a"` values, an all-empty column, duplicate rows) so
  `clean()` and `report()` have something to demonstrate.
  Root cause: `.gitignore` ignored `*.csv` with an exception only for
  `tests/data/`, so `git add` silently skipped the file. `dclean/data/*.csv` is
  now excepted too, which is what let the file go missing for four releases.
- `describe()` no longer raises `KeyError: 'mean'` on frames with no numeric
  columns; it falls back to pandas' object summary.
- `to_float()` strips currency symbols, thousands separators and parenthesized
  negatives before coercing, so `"$1,234.50"` and `"(500)"` parse correctly
  instead of becoming NaN.
- `filter()`'s `query()` fallback returned a DataFrame where the caller expected
  a boolean mask; the result is now handled correctly either way.
- Bundled-sample lookup used `os.sep`, which missed `/`-style paths on Windows;
  it now uses `os.path.dirname`.
- `mutate()` accepts accessor expressions such as
  `mutate(city="city.str.strip().str.lower()")`; `DataFrame.eval` cannot parse
  those, and it previously raised `ValueError: unknown type object`, forcing a
  `.to_df()` escape for any string operation.
- `drop_outliers(method="zscore")` dropped **every** row when a column had no
  spread (a constant column gave `std == 0`, which the guard treated as "keep
  nothing"). Columns with no spread are now skipped - they contain no outliers.
- Whole-frame stats (`mean()`, `sum()`, ... with no column) raised `TypeError`
  when any text column was present; they now aggregate numeric columns only.
- `plot()`'s inferred axes put the only numeric column on `x` for a one-column
  frame, leaving nothing to chart; `y` is now always the value column. A frame
  with no numeric column raises an explanatory error instead of a bare pandas
  `TypeError`.
- `fix_nulls()` silently fell back to the median on an unrecognized strategy;
  it now raises `ValueError` listing the valid ones.
- **`clean()` no longer destroys values silently.** A column converts to numeric
  when >90% of its values parse, so up to 10% of real values could become NaN
  with no mention of it — on a column of amounts holding "pending"/"refunded",
  those values simply disappeared. Both the numeric and date coercion steps now
  count and report what they could not parse.
- `report(examples=False)` shows value *types* instead of real cell values, for
  profiling data containing anything personal without echoing it to stdout.
- `filter()` / `mutate()` reject expressions containing `__`. These methods
  execute their argument, so the string must come from the developer; this
  blocks the usual escape out of a restricted namespace. A guard rail, not a
  sandbox — see the README's "Using it on real data".
- Replaced a `str.removeprefix` call, which is Python 3.9+ and would have failed
  the 3.8 leg of CI.
- `nulls()` now reports a percentage alongside each count.

### Docs
- README gains a **Command reference** section listing every command grouped by
  task, and a **Transforms never mutate** section explaining the return-value
  rule.
- All three example notebooks rewritten against the bundled dataset (they used
  to synthesize their own) and updated for the new API. `with_dcleaner.ipynb`
  and `without_dcleaner.ipynb` now perform the *same* cleaning work, so the
  comparison is honest: one call versus ~25 lines, verified to produce
  identical frames.

## [0.1.6] - 2026-07-19

### Added
- `dtypes()` — print each feature's data type.
- `print([n])` — print the dataset itself, chainable.
- `shape()` now also prints the column list and per-column dtypes.
- `print(d)` renders the dataset table (`__str__`); `repr(d)` stays terse.

## [0.1.5] - 2026-07-19

### Added
- `to_table([max_rows])` — render the FULL dataset as a tidy GitHub-style
  table, so you can print the whole frame (not just a slice) on screen.
- `nulls([plot])` — print missing-value counts per column plus the grand
  total; pass `plot=True` to also render a bar chart of the null counts.
- `to_float(*cols)` — convert string/object column(s) to float (unparseable
  values become NaN). With no args, every object column is coerced.

### Changed
- `head()` / `tail()` now render clean GitHub-style tables via `tabulate`
  with **all columns shown** (pandas' `...` truncation is gone — both the
  `display.max_columns` option and tabulate ensure every column prints).
- `describe()` now prints a highlighted banner and a tidy table, and calls out
  the **mean** per column so the headline statistic jumps out at a glance.
- `print(d)` / `repr(d)` now shows shape AND the column list
  (e.g. `dclean.Data(60×4, cols=[city, age, salary, score])`) instead of just
  the shape, so it no longer duplicates `shape()`'s output.
- Added `tabulate>=0.8` as a dependency for table rendering.

### Fixed
- README/code parity: every documented method now ships and works as written.

## [0.1.3] - 2026-07-10

### Added
- Bundled sample dataset `sample_sales.csv` ships inside the package. A bare
  filename that isn't found on disk is resolved against the bundled data, so
  `Data("sample_sales.csv")` works immediately after `pip install dcleaner` on
  any machine — no manual file downloads.

## [0.1.2] - 2026-07-10

### Fixed
- `keep(*cols)` now accepts multiple positional column names (e.g.
  `.keep("name", "price")`), matching `select(*cols)` — previously required a
  list.
- `filter("x between lo and hi")` is now supported (SQL-style `between`),
  rewritten to `x >= lo and x <= hi`.
- Spaced column names in `filter()` documented with backticks
  (`` d.filter("`total sales` > 100") ``); single quotes compared a string
  literal and raised. Single quotes already worked for string equality on
  normal columns (e.g. `city == 'NY'`).

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
