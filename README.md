# dcleaner

**Fluent, low-boilerplate data cleaning and visualization — built on top of pandas.**

Tired of writing the same rename-strip-coerce-dedupe boilerplate every time you
open a file? `dcleaner` is a **toolkit**: tell it what you want and it works out
the rest — `dclean.clean("messy.csv", to="clean.csv")` normalizes the column
names, strips the `$` and commas off your numbers, parses the dates, drops the
empty columns and kills the duplicates, then tells you exactly what it did.

When you want the control instead, the same operations are a chainable, readable
API over pandas, so a full **load → clean → filter → aggregate → plot** pipeline
is one expression. Transforms never mutate what you pass them, and `.to_df()`
drops you back into full pandas whenever you outgrow the wrapper.

![CI](https://github.com/Shivmahlan/dcleaner/actions/workflows/ci.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/dcleaner.svg)](https://pypi.org/project/dcleaner/)
[![OpenSSF Best Practices](https://img.shields.io/badge/OpenSSF-baseline-blue)](https://www.bestpractices.dev/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shivmahlan/dcleaner/blob/main/examples/demo.ipynb)

---

## Table of contents

- [Install](#install)
- [The idea](#the-idea)
- [Quick start](#quick-start)
- [The toolkit: just tell it](#the-toolkit-just-tell-it)
- [From the shell](#from-the-shell)
- [Transforms never mutate](#transforms-never-mutate)
- [Loading data](#loading-data)
- [Inspecting](#inspecting)
- [Cleaning](#cleaning)
- [Filtering](#filtering)
- [Transforming](#transforming)
- [Combining tables](#combining-tables)
- [Aggregating](#aggregating)
- [Correlations](#correlations)
- [Plotting](#plotting)
- [Exporting](#exporting)
- [Recipes](#recipes)
- [Command reference](#command-reference)
- [Full API reference](#full-api-reference)
- [Why not just use pandas?](#why-not-just-use-pandas)
- [Design notes](#design-notes)
- [Using it on real data](#using-it-on-real-data)
- [License](#license)

---

## Install

```bash
pip install dcleaner
```

Requirements: Python ≥ 3.8, plus `pandas`, `matplotlib`, and `numpy`
(automatically installed as dependencies).

## Try it

A messy sample dataset ships inside the package, so every example below runs
as-is — no downloads, no setup:

```python
import dclean
dclean.report("sample_sales.csv")     # see what's wrong with it
dclean.clean("sample_sales.csv")      # fix it
```

[`examples/demo.ipynb`](examples/demo.ipynb) walks the whole toolkit:
profile → clean → aggregate → plot. Run it without installing anything:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shivmahlan/dcleaner/blob/main/examples/demo.ipynb)

See the time saved side-by-side:
[`examples/with_dcleaner.ipynb`](examples/with_dcleaner.ipynb) vs
[`examples/without_dcleaner.ipynb`](examples/without_dcleaner.ipynb) — the same
cleaning job, one call versus ~25 lines of pandas, producing byte-identical
results.

---

## The idea

Most EDA follows the same shape:

1. Load a file.
2. Drop missing rows, fix types, rename ugly columns.
3. Filter to the rows you care about.
4. Summarize / group / correlate.
5. Plot something to check your intuition.

With raw pandas that's a pile of intermediate variables and method calls.
With `dclean` you either hand the whole job to `.clean()`, or chain the steps —
each one returns a new `Data`, so the pipeline reads top to bottom:

```python
from dclean import Data

# A sample dataset ships with the library, so this works right after
# `pip install dcleaner` — no file downloads needed:
result = (Data("sample_sales.csv")
    .clean()
    .filter("units > 5 and city in ['SF', 'Chicago']")
    .groupby("city").agg("mean", "unit_price")
    .plot("bar", x="city", y="unit_price", title="Mean price by city")
    .show())          # the chart opens right here - no PNG to go and find
```

The `.clean()` step alone is the part you would otherwise write by hand every
single time: renaming columns, stripping `$` and commas, parsing dates,
dropping the empty column, killing duplicates.

---

## Quick start

```python
from dclean import Data

# Load, clean, filter, aggregate, plot — in one chain.
(Data("sales.csv")
    .dropna()
    .filter("age > 18")
    .groupby("city").agg("mean", "salary")
    .plot("bar", x="city", y="salary")
    .show())
```

---

## The toolkit: just tell it

You should not have to hand-write the same fifteen cleaning steps for every
file. Say what you want; the library works out the rest.

```python
import dclean

dclean.clean("messy.csv", to="clean.csv")   # load, clean, save. One line.
dclean.report("messy.csv")                  # full data-quality profile
```

`clean()` inspects the data and applies what it needs:

1. normalizes column names — `" Total Sales ($) "` → `total_sales`
2. drops all-empty columns and all-empty rows
3. trims whitespace on text, and turns `""` / `"n/a"` / `"null"` / `"-"` into real NaN
4. converts text that is really numbers to numbers — **currency and thousands
   separators included**: `"$1,234.50"` → `1234.5`, `"(500)"` → `-500`
5. parses date-looking columns into real datetimes
6. removes duplicate rows
7. handles the remaining nulls the way you asked

It prints exactly what it did, so nothing happens behind your back:

```
CLEAN
  + normalized 9 column names
  + trimmed whitespace on 7 text columns
  + dropped 1 empty columns: legacy_column
  + converted to numeric: unit_price
  + parsed as dates: order_date
  + removed 4 duplicate rows
-> 64x9 to 60x8, 56 nulls remaining
```

Missing values are yours to direct:

```python
d.clean()                  # nulls="keep"  - leave them (default)
d.clean(nulls="drop")      # drop any row with a null
d.clean(nulls="fill")      # median for numeric, most-common for text
d.clean(verbose=False)     # do it silently
```

### Know your data before you touch it

`report()` is the "what am I even looking at" button — per-column dtypes, null
counts and percentages, unique counts, example values, duplicate-row count,
numeric summary stats, and a list of problems worth your attention:

```python
Data("sales.csv").report()
```

```
warnings
  ! 4 duplicate rows - .clean() or .dedupe() removes them
  ! 'Unit Price' looks numeric but is stored as text - .clean() fixes it
  ! 'Notes' is 88% missing
  ! 'Legacy Column' is entirely empty
```

**Send it to someone.** `to=` writes the same profile as a single HTML file —
styles inline, nothing loaded from a CDN — so it opens anywhere: attached to a
ticket, emailed, or dropped in a shared folder.

```python
Data("sales.csv").report(to="profile.html")
Data("sales.csv").report(examples=False, to="profile.html")   # no real values
```

`examples=False` matters more here than in the terminal: the HTML file is the
one somebody else ends up holding, so that flag keeps your actual data out of
it — the column shows `<str>` / `<int64>` instead.

### One call per idea

Common questions shouldn't cost you two calls and a dummy column:

```python
d.mean("price", by="city")      # instead of .groupby("city").agg("mean","price")
d.count(by="city")              # rows per group - no dummy column
d.sum("revenue", by=["city","year"])
d.top(5, "price")               # instead of .sort("price", ascending=False).head(5)
d.bottom(5, "price")
d.counts("city")                # frequency table - no .to_df() escape
d.median("price"); d.min("price"); d.max("price")   # whole-frame too
```

`x` and `y` are inferred when they're obvious, so a grouped aggregate plots
straight away:

```python
(Data("sales.csv").clean().mean("price", by="city").plot("bar").show())
```

The longhand `.groupby().agg()` still works and returns identical results —
these are shortcuts, not replacements.

### The rest of the toolkit

```python
d.fix_nulls()                       # median for numeric, mode for text
d.fix_nulls(strategy="ffill")       # or "mean"/"median"/"mode"/"zero"
d.fix_nulls(subset=["price"])       # only these columns
d.drop_outliers()                   # IQR rule across all numeric columns
d.drop_outliers("price", method="zscore", factor=3)
d.log()                             # print every step that produced this frame
d.steps()                           # ...or get them back as a list
```

`log()` matters because the toolkit makes decisions for you — it keeps the
pipeline auditable:

```python
(Data("sales.csv").clean(verbose=False).filter("units > 5").sort("units").log())
# pipeline
#   1. clean(nulls='keep')
#   2. filter('units > 5')
#   3. sort('units')
```

### A dataset ships with the library

Every example here runs immediately after `pip install dcleaner` — no
downloads:

```python
Data.samples()                  # -> ['sample_sales.csv']
dclean.report("sample_sales.csv")
```

It is deliberately messy — padded column names, `$1,234.50` prices, `"n/a"`
units, an empty column, duplicate rows — so `clean()` has something to do.

---

## From the shell

Some jobs never need a Python session. `pip install dcleaner` puts a `dcleaner`
command on your PATH, and every subcommand is a thin wrapper over the API of
the same name — so the terminal and the library can never disagree about what
"clean" means.

```bash
dcleaner report sales.csv                        # full data-quality profile
dcleaner report sales.csv --no-examples          # ...with no real values in it
dcleaner report sales.csv --html profile.html    # ...as a shareable HTML file

dcleaner clean messy.csv                         # clean it, say what changed
dcleaner clean messy.csv -o clean.csv            # ...and write it out
dcleaner clean messy.csv -o clean.parquet --nulls fill -q

dcleaner head sales.csv -n 20                    # peek at the first 20 rows
dcleaner --version
```

| Command | Flags |
|---|---|
| `dcleaner report FILE` | `--no-examples` (show types, not your values) · `--html OUT` (self-contained file) |
| `dcleaner clean FILE` | `-o/--out OUT` (csv/xlsx/json/parquet) · `--nulls keep\|drop\|fill` · `--no-dates` · `-q/--quiet` |
| `dcleaner head FILE` | `-n N` (default 5) |

It behaves like a shell tool should: **exit 0** on success, **exit 1** with one
line on stderr for anything you can fix — a missing file, a file type `dclean`
doesn't read, an output extension it can't write. A traceback means a bug in
`dclean`, not a typo in a filename.

```bash
$ dcleaner report nope.csv
dcleaner: no such file: nope.csv
$ echo $?
1
```

Piped output is plain text — the bold/underline escapes are dropped
automatically when stdout isn't a terminal, so `dcleaner report x.csv > profile.txt`
gives you a clean file. `NO_COLOR` and `FORCE_COLOR` are both respected.

---

## Transforms never mutate

**Every transform returns a new `Data`. The object you called it on is never
modified.**

```python
raw = Data("sales.csv")
clean = raw.dropna()

len(raw)      # unchanged - every original row is still here
len(clean)    # the cleaned copy
```

This is the pandas contract (`df.dropna()` doesn't touch `df`) and it means
intermediate variables keep the value you assigned them. Chains work exactly
as before, because each step passes its new object to the next:

```python
(Data("sales.csv")
    .clean()
    .filter("units > 5")
    .groupby("city").agg("mean", "unit_price")
    .plot("bar", x="city", y="unit_price")
    .show())
```

The rule is simple:

| Kind of method | Returns |
|---|---|
| Transforms — `clean` `dropna` `filter` `mutate` `agg` `join` `concat` … | a **new** `Data` |
| Inspectors — `head` `report` `nulls` `describe` `log` … | the **same** object (they change nothing) |
| Plot / output — `plot` `savefig` `to_csv` `to_excel` `to_json` `to_parquet` `save_recipe` `show` | the **same** object |
| Escape hatches — `to_df` `to_fig` `steps` `len` `repr` | a plain value |

> **Upgrading from 0.1.x?** Transforms used to modify in place, so
> `d.to_float("price")` worked as a statement. Now you must bind the result:
> `d = d.to_float("price")`. Anything already written as a single chain is
> unaffected.

---

## Loading data

`Data()` auto-detects the format from the file extension.

```python
Data("data.csv")          # CSV (also .csv.gz)
Data("data.xlsx")         # Excel (.xls / .xlsx)
Data("data.json")         # JSON
Data("data.parquet")      # Parquet

# Already have a DataFrame? Pass it directly:
import pandas as pd
Data(pd.read_csv("data.csv"))

# Or build from a list of dicts:
Data.from_records([{"name": "a", "val": 1}, {"name": "b", "val": 2}])
```

---

## Inspecting

```python
d = Data("sales.csv")
d.head()            # first 5 rows
d.head(10)          # first 10 rows
d.tail()            # last 5 rows
d.print()           # print the FULL dataset (chainable)
d.print(10)         # print first 10 rows
d.to_table()        # print the FULL dataset as a tidy table
print(d)            # prints the dataset table (str); repr stays terse
d.shape()           # prints "N rows x M cols" + columns + dtypes
d.dtypes()          # prints each feature's dtype (column -> type)
d.cols()            # prints the column list
d.info()            # pandas .info()  (incl. per-column dtypes)
d.describe()        # highlighted summary stats of numeric columns
d.nulls()           # missing-value counts per column (+ total)
d.report()          # FULL profile: dtypes, nulls, dupes, stats, warnings
d.report(to="p.html")  # ...the same profile as one shareable HTML file
d.log()             # the pipeline steps that produced this frame
```

---

## Cleaning

Reach for `.clean()` first — it does the whole list below, automatically:

```python
Data("sales.csv").clean()
```

Everything is also available individually when you want the control:

```python
(Data("sales.csv")
    .dropna()                 # drop any row with a missing value
    .dropna(subset=["price"]) # drop only rows missing 'price'
    .fillna(0)                # replace all NaN with 0
    .fillna({"age": 0, "city": "unknown"})  # per-column fill
    .dedupe()                 # drop duplicate rows
    .dedupe(subset=["id"])    # dedupe on a key
    .drop("notes")            # remove a column
    .keep("name", "price")    # keep ONLY these columns
    .rename(price="cost")     # rename a column
    .lower_cols()             # lowercase ALL column names (great first step)
    .astype(price="float")    # cast types
    .to_float("price", "qty") # convert string/object cols to float (errors→NaN)
    .fix_nulls()              # median for numeric, most-common for text
    .drop_outliers()          # IQR rule on numeric columns
    .nulls())                 # show missing-value counts per column
```

> Tip: start every pipeline with `.lower_cols()` so you never have to remember
> whether a column is `Price`, `price`, or `PRICE`.

To spot dirty data fast, chain `.nulls()` right before `.dropna()` — it prints
a per-column null count and the grand total:

```python
(Data("sales.csv").nulls())   # → total nulls: N across M rows
```

---

## Filtering

`filter()` takes a plain expression string — no lambdas, no bracket soup.

```python
d.filter("age > 18")
d.filter("city == 'NY'")
d.filter("age > 18 and city in ['NY', 'LA']")
d.filter("salary >= 50000 or department == 'eng'")
d.filter("status != 'inactive'")
d.filter("score between 70 and 100")   # pandas eval supports between
```

Columns whose names contain spaces must use backticks (standard
`pandas.eval()` / `.query()` syntax):

```python
d.filter("`total sales` > 100")
```

Under the hood `filter()` uses `DataFrame.eval()` (falling back to
`DataFrame.query()`), so it stays fast and vectorized.

---

## Transforming

`mutate()` adds or overwrites columns from expressions — again, no lambdas.

```python
(Data("people.csv")
    .mutate(bmi="weight / (height**2)")
    .mutate(age_plus_1="age + 1")
    .mutate(is_adult="age >= 18"))     # 0/1 boolean column

# String accessors work too:
d.mutate(city="city.str.strip().str.lower()")

# Pass a literal (non-string) value to assign it directly:
d.mutate(flag=True)
```

Other transforms:

```python
d.select("name", "price")   # keep only these columns
d.keep("name", "price")     # same, but ignores non-string args
d.sort("price")             # ascending
d.sort("price", ascending=False)
```

---

## Combining tables

### `join()` — attach another table, and see what actually matched

```python
d.join("regions.csv", on="city")                  # left join (the default)
d.join(other, on=["city", "year"], how="inner")   # multi-column key
d.join(other, left_on="city_id", right_on="id")   # the two sides spell it differently
```

`other` is a `Data`, a DataFrame or a path. `on` defaults to the columns the two
tables share. `how` is `left` (default), `inner`, `right` or `outer`.

**The report is the feature.** A join where one side says `"SF "` and the other
says `"SF"` matches nothing, tells you nothing, and quietly poisons every number
downstream. So `join()` counts what matched before it merges:

```
JOIN
  + left join on 'city'
  + 2 of 4 left rows matched, 2 of 5 right rows used
  ! 2 rows on the left matched nothing (kept, with nulls): 'SF ', 'chicago'
  ! 2 of those keys match after case/whitespace folding - .clean() both sides before joining
  ! 3 rows on the right matched nothing (dropped)
  + right columns kept under a suffix: notes_right
-> 4x3 to 4x5
```

Every line there is a bug people actually ship: keys that miss, keys that would
match after a `.clean()`, rows silently dropped by an `inner` join, and — the
one that inflates totals without changing the row count you were watching — a
non-unique key on the right:

```
  ! the right key is not unique - the join added 340 rows
```

A join that *cannot* work is refused up front rather than returning an empty
frame:

```python
d.join(other, on="id")
# ValueError: join key 'id' is text on the left but 'id' is number on the right
# - they can never match. Make them agree first, e.g. .to_float('id')
```

Overlapping non-key columns keep their name on the left; the right-hand one
gets a suffix (`suffix="_right"` by default). `verbose=False` skips the report —
and the counting work with it, so a join inside a loop costs no more than
`pd.merge`.

### `concat()` — stack tables on top of each other

```python
Data.concat("data/2024-*.csv")                       # a whole folder of files
Data.concat("jan.csv", "feb.csv", source_col="file") # tag each row with its file
Data.concat([d1, d2, df3])                           # Data, DataFrames, paths
d.concat(more)                                       # onto an existing Data
```

Columns are matched by name, and the report calls out the two things that
quietly ruin a stacked dataset — a column missing from one source (it silently
becomes nulls) and a column whose type changes between sources:

```
CONCAT
  + 2 sources: m_2024-01.csv (1 row), m_2024-02.csv (1 row)
  ! 'notes' is missing from m_2024-02.csv - those rows are null there
  ! 'extra' is missing from m_2024-01.csv - those rows are null there
  + tagged every row with its source in 'file'
-> 2x5
```

`source_col` is worth the extra column the moment two files disagree — it is the
difference between "some rows are wrong" and "the February export is wrong".

### The whole job, end to end

```python
(Data.concat("exports/*.csv", source_col="file")   # every monthly export
    .clean()                                       # names, types, dates, dupes
    .join("regions.csv", on="city")                # attach the lookup table
    .filter("units > 5")
    .sum("revenue", by="region")
    .plot("bar", title="Revenue by region")
    .show())
```

---

## Aggregating

Group then aggregate. `agg(how, col)` computes one statistic on one column.

```python
# Mean salary per city
(Data("sales.csv")
    .groupby("city").agg("mean", "salary"))

# Count of rows per department
(Data("sales.csv")
    .groupby("department").agg("count", "id"))

# Multiple groups
(Data("sales.csv")
    .groupby("city", "year").agg("sum", "revenue"))
```

Supported `how` values are any pandas aggregation: `mean`, `sum`, `count`,
`min`, `max`, `median`, `std`, etc.

`summarize()` computes several named stats at once into a one-row frame:

```python
(Data("sales.csv")
    .summarize(mean_salary="mean(salary)",
               max_salary="max(salary)",
               n="count()"))
```

---

## Correlations

```python
# Get the correlation matrix as a DataFrame (handy for further work)
corr_df = Data("sales.csv").dropna().corr().to_df()
print(corr_df)

# Or plot it directly as a heatmap
(Data("sales.csv")
    .dropna()
    .plot_corr(title="Feature correlations")
    .show())
```

`corr()` and `plot_corr()` use Pearson correlation on numeric columns only.

---

## Plotting

`plot(kind, ...)` supports the common chart types. `x` and `y` name the
columns; `title` sets the title.

```python
d = Data("sales.csv").dropna()

d.plot("line",   x="date",   y="revenue")              # line chart
d.plot("bar",    x="city",   y="salary")               # bar chart
d.plot("hist",   x="age")                              # histogram
d.plot("scatter",x="age",    y="salary")               # scatter
d.plot("box",    x="city",   y="salary")               # boxplot
d.plot("pie",    x="city",   y="salary")               # pie chart
```

### Where the chart appears

The plot shows up where you are working — you never have to open a PNG to see
what you just plotted:

| Where you run it | What happens |
|---|---|
| Jupyter / IPython notebook | renders **inline**, in the same cell, as soon as you call `.plot()` |
| Script or REPL | end the chain with `.show()` (or pass `show=True`) and it opens in a window |
| No display at all (CI, a server over SSH) | `.savefig("chart.png")` writes the file; `.show()` tells you there is no display instead of silently doing nothing |

```python
# notebook - the bar chart appears under the cell, no savefig needed
(Data("sales.csv")
    .dropna()
    .groupby("city").agg("mean", "salary")
    .plot("bar", x="city", y="salary", title="Mean salary by city"))

# script - .show() opens the window
(Data("sales.csv")
    .dropna()
    .groupby("city").agg("mean", "salary")
    .plot("bar", x="city", y="salary", title="Mean salary by city")
    .show())

# want the file as well? savefig still does exactly what it always did
d.plot("bar", x="city", y="salary").savefig("salary_by_city.png")
```

`show=` overrides the automatic choice on `plot()`, `plot_corr()` and
`nulls(plot=True)`:

```python
d.plot("bar", show=True)     # display it now, wherever I am
d.plot("bar", show=False)    # display nothing; I only want .savefig()
```

`.to_fig()` hands back the matplotlib `Figure` — the plotting equivalent of
`.to_df()` — for anything `dclean` doesn't wrap:

```python
fig = d.plot("bar").to_fig()
fig.axes[0].set_ylabel("mean salary ($)")
```

Extra matplotlib keyword arguments pass straight through:

```python
d.plot("scatter", x="age", y="salary", color="red", alpha=0.5)
```

> Note: `dclean` does not pin a matplotlib backend, so plots render inline in
> notebooks and in a window from a script. Matplotlib's own detection still
> falls back to a file-only backend when there is no display, so `savefig()`
> keeps working in scripts, CI and servers.

---

## Exporting

`dclean` writes back every format it reads:

```python
d.to_csv("cleaned.csv")          # write the current frame, no index
d.to_excel("cleaned.xlsx")       # needs: pip install openpyxl
d.to_json("cleaned.json")        # records + ISO-8601 dates by default
d.to_parquet("cleaned.parquet")  # needs: pip install pyarrow
```

All four return the **same** object, so an export can sit mid-chain:

```python
(Data("messy.csv")
 .clean()
 .to_csv("clean.csv")            # snapshot the cleaned data...
 .filter("units > 5")            # ...and carry on
 .to_parquet("big_orders.parquet"))
```

`openpyxl` and `pyarrow` are **not** installed with `dcleaner` — they are
imported only when you call the method that needs one, and the error tells you
the exact `pip install` if it is missing.

```python
raw = d.to_df()                  # get the raw pandas DataFrame back
raw.describe()                   # now use any pandas method you like
```

`.to_df()` is the escape hatch: `dclean` never hides pandas from you. Use it
for anything `dclean` doesn't wrap yet.

---

## Recipes

You cleaned January. February is the same file with different rows — and next
month there'll be another one. A **recipe** saves what your pipeline did so you
can run it again on a different file:

```python
Data("jan.csv").clean().filter("units > 5").save_recipe("monthly.json")
Data("feb.csv").apply_recipe("monthly.json")
```

`save_recipe()` is an output method (writes the file, returns the same object);
`apply_recipe()` is a transform (returns a **new** `Data`). The replayed steps
land in the new object's own `.log()`, so provenance survives the round trip:

```python
Data("feb.csv").apply_recipe("monthly.json").log()
```
```
pipeline
  1. clean(nulls='keep')
  2. filter('units > 5')
```

The file is plain, reviewable JSON — it belongs in git next to the code that
uses it:

```json
{
  "recipe": 1,
  "dcleaner": "0.3.0",
  "created": "2026-08-30T12:58:29",
  "steps": [
    {"method": "clean",  "args": [],              "kwargs": {"nulls": "keep", "dates": true},
     "display": "clean(nulls='keep')"},
    {"method": "filter", "args": ["units > 5"],   "kwargs": {},
     "display": "filter('units > 5')"}
  ]
}
```

`display` is there for you to read in a diff — **replay never parses it**, only
`method`/`args`/`kwargs`. That's the whole reason steps are recorded
structurally instead of being reverse-engineered from the log: a column named
`it's (a) column` would defeat any parser of dclean's own repr.

**Not everything replays.** The whitelist is the transforms that are a pure
function of the frame plus plain arguments. `join()` and `concat()` need a
second table a recipe can't carry, and `groupby().agg()` is two calls — so
`save_recipe()` **refuses** rather than quietly dropping the step:

```python
d.groupby("city").agg("mean", "salary").save_recipe("r.json")
# ValueError: step 2 (groupby(['city']).agg('mean', 'salary')) cannot be
# replayed - groupby().agg() is two calls - use .mean(col, by=...) ...
```

A recipe that silently skipped your join would hand you a different dataset
without saying so. Use the one-call form (`.mean("salary", by="city")`), which
records as a single step, or save the recipe before the join.

> ⚠️ **A recipe is code.** `filter()` and `mutate()` steps are expressions that
> get evaluated on replay, exactly as if you had typed them. A recipe file is
> as trusted as a Python script — run the ones you wrote, not ones you were
> sent. The whitelist is checked again at replay time (the file may have been
> edited), so an unknown method is an error, never a silent skip.

---

## Command reference

Every command, grouped by what you're trying to do. All of them chain.

### Load

```python
from dclean import Data
import dclean

Data("sales.csv")                    # csv, csv.gz, xls, xlsx, json, parquet
Data(df)                             # wrap an existing DataFrame
Data.from_records([{"a": 1}])        # from a list of dicts
Data.samples()                       # list datasets bundled with the package
dclean.load("sales.csv")             # same as Data(...)
```

### Do the whole job in one call

```python
dclean.clean("messy.csv")                    # load + clean
dclean.clean("messy.csv", to="clean.csv")    # load + clean + save
dclean.clean("messy.csv", nulls="fill")      # ...and fill the gaps
dclean.report("messy.csv")                   # full data-quality profile
```

### Inspect

```python
d.head()          d.head(10)         # first rows
d.tail()          d.tail(10)         # last rows
d.print()         d.print(10)        # the dataset itself, chainable
d.to_table()      d.to_table(50)     # full table, every column
d.shape()                            # rows x cols + columns + dtypes
d.dtypes()                           # column -> dtype
d.cols()                             # column list
d.info()                             # pandas .info()
d.describe()                         # highlighted numeric summary
d.nulls()         d.nulls(plot=True) # missing values per column (+ chart)
d.report()                           # dtypes, nulls, dupes, stats, warnings
d.report(to="profile.html")          # same profile, one self-contained file
d.log()           d.steps()          # what this pipeline actually did
d.save_recipe("r.json")              # save those steps...
Data("feb.csv").apply_recipe("r.json")   # ...and replay them elsewhere
print(d)          len(d)             # table render / row count
```

### Clean

```python
d.clean()                            # do everything below, automatically
d.clean(nulls="drop")                # ...and drop rows with nulls
d.clean(nulls="fill")                # ...median for numbers, mode for text
d.clean(dates=False)                 # ...but leave date-looking text alone
d.clean(verbose=False)               # ...silently

d.dropna()        d.dropna(subset=["price"])
d.fillna(0)       d.fillna({"age": 0, "city": "unknown"})
d.fix_nulls()                        # auto: median numeric, mode text
d.fix_nulls("ffill")                 # or "mean"/"median"/"mode"/"zero"
d.fix_nulls("median", subset=["price"])
d.dedupe()        d.dedupe(subset=["id"])
d.drop("notes")   d.drop(["notes", "tmp"])
d.keep("name", "price")              # keep only these
d.rename(price="cost")
d.lower_cols()                       # lowercase every column name
d.astype(price="float")
d.to_float("price")   d.to_float()   # "$1,234.50" -> 1234.5
d.drop_outliers()                    # IQR rule, all numeric columns
d.drop_outliers("price", method="zscore", factor=3)
```

### Filter and transform

```python
d.filter("age > 18")
d.filter("city == 'NY'")
d.filter("age > 18 and city in ['NY', 'LA']")
d.filter("score between 70 and 100")
d.filter("`total sales` > 100")          # backticks for spaced names

d.mutate(revenue="units * price")        # arithmetic
d.mutate(city="city.str.strip().str.lower()")   # string accessors
d.mutate(flag=True)                      # literal value
d.select("name", "price")
d.sort("price")   d.sort("price", ascending=False)
```

### Combine

```python
d.join("regions.csv", on="city")     # left join + a report of what matched
d.join(other, on="city", how="inner")            # left/inner/right/outer
d.join(other, on=["city", "year"])               # multi-column key
d.join(other, left_on="city_id", right_on="id")  # differently named keys
d.join(other, on="city", suffix="_lookup")       # rename right-hand clashes
d.join(other, on="city", verbose=False)          # skip the report

Data.concat("data/*.csv")            # stack a folder of files
Data.concat("a.csv", "b.csv", source_col="file") # ...and record where each row came from
d.concat(other)                      # stack onto an existing Data
```

### Aggregate

```python
d.mean("price", by="city")           # one call, no separate groupby
d.sum("revenue", by=["city", "year"])
d.count(by="city")                   # rows per group
d.median("price")  d.min("price")  d.max("price")
d.top(5, "price")     d.bottom(5, "price")
d.counts("city")                     # frequency table
d.corr()                             # correlation matrix

d.groupby("city").agg("mean", "price")           # longhand, same result
d.summarize(avg="mean(price)", n="count()")
d.stat("std", "price", by="city")                # any pandas aggregation
```

### Plot and export

```python
d.plot("bar")                        # x/y inferred on a 2-column frame
d.plot("bar", x="city", y="price", title="Mean price")
d.plot("line", x="date", y="revenue")
d.plot("hist", x="age")
d.plot("scatter", x="age", y="salary", color="red", alpha=0.5)
d.plot("box", x="city", y="salary")
d.plot("pie", x="city", y="salary")
d.plot_corr(title="Feature correlations")
d.plot("bar", show=False)            # build the figure, display nothing
d.show()                             # inline in a notebook, a window from a script
d.savefig("chart.png")               # write it to a file (works headless)

d.to_csv("clean.csv")
d.to_df()                            # the raw DataFrame - full pandas
d.to_fig()                           # the matplotlib Figure - full matplotlib
```

---

## Full API reference

| Task | Method | Notes |
|------|--------|-------|
| **Do it all** | `dclean.clean(src, to=...)` `dclean.report(src)` `.clean([nulls])` `.report([examples], [to])` | one call: name the file, the library handles the rest; `report(to="x.html")` writes a shareable profile |
| Load file | `Data("file.csv")` | auto-detects csv/xls/xlsx/json/parquet |
| From frame | `Data(df)` / `Data.from_records([...])` | |
| Inspect | `.head(n)` `.tail(n)` `.print([n])` `.to_table([max_rows])` `.shape()` `.dtypes()` `.cols()` `.info()` `.describe()` `.nulls([plot], [show])` | `print(d)` renders the dataset; `repr(d)` stays terse. `.shape()` and `.dtypes()` show per-feature types |
| Clean | `.clean([nulls])` `.dropna([subset])` `.fillna(v)` `.fix_nulls([strategy])` `.drop_outliers([cols])` `.dedupe([subset])` `.drop(c)` `.keep(*c)` `.rename(a=b)` `.lower_cols()` `.astype(a="t")` `.to_float(*cols)` | `.to_float()` strips currency/separators, unparseable→NaN |
| Filter | `.filter("expr")` | `== != > < >= <= and or in not in` + `between` |
| Transform | `.mutate(x="expr")` `.select(*c)` `.sort(by, [ascending])` | |
| Combine | `.join(other, [on], [how], [left_on], [right_on], [suffix])` `Data.concat(*sources, [source_col])` | reports matched/unmatched rows, near-miss keys, row multiplication, column and type mismatches |
| Aggregate | `.mean/.sum/.count/.median/.min/.max([col], [by])` `.top(n, by)` `.bottom(n, by)` `.counts(col)` `.stat(how, col, by)` | one call, no separate `groupby` step |
| Aggregate (longhand) | `.groupby(*c).agg(how, col)` `.summarize(**stats)` `.corr([method])` | |
| Plot | `.plot(kind, [x], [y], [title], [show])` `.plot_corr([title], [show])` | line/bar/hist/scatter/box/pie; `x`/`y` inferred on a 2-column frame; `show=` overrides where it renders |
| Output | `.show()` `.savefig(path)` `.to_csv(path)` `.to_excel(path)` `.to_json(path)` `.to_parquet(path)` `.to_df()` `.to_fig()` | `.show()`/`.plot()` render inline in notebooks, in a window from a script; `to_excel`/`to_parquet` need `openpyxl`/`pyarrow` |
| Provenance | `.log()` `.steps()` `.save_recipe(path)` `.apply_recipe(path)` | what the pipeline did — and running it again on another file |

Transform methods return a **new** `Data` (the original is untouched);
inspect, plot and output methods return the **same** object; `.to_df()`,
`.steps()`, `len(d)` and `repr(d)` hand back plain values. Either way they all
chain. `head()`/`tail()`/`to_table()` render clean GitHub-style tables with
**every column shown** (no `...` truncation).

---

## Why not just use pandas?

You are — `dclean` is pandas underneath. The value is:

- **Less boilerplate** for the 90% case (quick EDA, one-off plots).
- **Readable pipelines** you can read top-to-bottom like a sentence.
- **No lambda gymnastics** for filters and derived columns.
- **A clean off-ramp**: `.to_df()` drops you into full pandas whenever you
  outgrow the wrapper.

It is not trying to replace pandas. It is trying to make the common path
shorter.

---

## Design notes

- **One call per idea.** `.clean()`, `.report()`, `.mean(by=...)`, `.top()` —
  the common jobs are a single call, not a recipe you reassemble each time.
- **Tell it, don't spell it.** `.clean()` and `.report()` decide what a file
  needs, do it, and print what they did.
- **Immutable.** Transforms return a new `Data`; nothing is modified under you.
- **Fluent by default.** Every operation returns a `Data`, enabling chains.
- **Auditable.** `.log()` replays every step the toolkit took on your behalf.
- **Vectorized.** `filter()` and `mutate()` use `DataFrame.eval`/`query`, so
  they stay fast on large frames — no Python-row loops.
- **Plots where you are.** No backend is pinned: charts render inline in a
  notebook and in a window from a script, and fall back to `savefig()` on a
  machine with no display — so seeing a plot never costs a file round-trip.
- **Escape hatch.** `.to_df()` gives you the raw DataFrame for anything not
  wrapped.

---

## Using it on real data

`dcleaner` runs entirely locally — no network calls, no telemetry, and it never
writes a file unless you call `to_csv()` / `savefig()`. Wrapping a DataFrame
copies it, and transforms are immutable, so your original frame is never
modified. Three things are still worth knowing.

**`clean()` tells you what it destroyed.** A text column converts to numbers
when >90% of its values parse — so up to 10% of real values can become `NaN`.
That is reported, never silent:

```
  + converted to numeric: amount
  ! 8 value(s) in 'amount' could not be parsed as numbers and became NaN
```

Run `.report()` first and `.log()` after, and check that line. For a production
or auditable pipeline, prefer the individual methods, which make no decisions
for you.

**`filter()` and `mutate()` execute their strings as code.** They are built on
`DataFrame.eval` / `query`. That is safe for expressions *you* write, but never
pass a string that came from an end user — a search box, a URL parameter, an
LLM's output. Expressions containing `__` are rejected, but that is a guard
rail, not a sandbox:

```python
d.filter(f"city == '{user_input}'")   # NO - user input becomes code
d.filter("city == @chosen_city")      # fine - the value is a variable
```

**`report()` prints real values** in its `example` column, which is exactly what
you want locally and exactly what you don't want in a shared log. Pass
`examples=False` to show types instead:

```python
d.report(examples=False)    # example column shows <str>, <int64>, ...
d.report(examples=False, to="profile.html")   # ...and in the shareable file
```

---

## License

MIT — see [LICENSE](LICENSE).
