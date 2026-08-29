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

---

## Table of contents

- [Install](#install)
- [The idea](#the-idea)
- [Quick start](#quick-start)
- [The toolkit: just tell it](#the-toolkit-just-tell-it)
- [Transforms never mutate](#transforms-never-mutate)
- [Loading data](#loading-data)
- [Inspecting](#inspecting)
- [Cleaning](#cleaning)
- [Filtering](#filtering)
- [Transforming](#transforming)
- [Aggregating](#aggregating)
- [Correlations](#correlations)
- [Plotting](#plotting)
- [Exporting](#exporting)
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
profile → clean → aggregate → plot.

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
    .savefig("price_by_city.png"))
```

The `.clean()` step alone is the part you would otherwise write by hand every
single time: renaming columns, stripping `$` and commas, parsing dates,
dropping the empty column, killing duplicates.

---

## Quick start

```python
from dclean import Data

# Load, clean, filter, aggregate, plot, save — in one chain.
(Data("sales.csv")
    .dropna()
    .filter("age > 18")
    .groupby("city").agg("mean", "salary")
    .plot("bar", x="city", y="salary")
    .savefig("salary.png"))
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
(Data("sales.csv").clean().mean("price", by="city").plot("bar").savefig("out.png"))
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
    .savefig("out.png"))
```

The rule is simple:

| Kind of method | Returns |
|---|---|
| Transforms — `clean` `dropna` `filter` `mutate` `agg` `to_float` … | a **new** `Data` |
| Inspectors — `head` `report` `nulls` `describe` `log` … | the **same** object (they change nothing) |
| Plot / output — `plot` `savefig` `to_csv` `show` | the **same** object |
| Escape hatches — `to_df` `steps` `len` `repr` | a plain value |

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
    .savefig("corr.png"))
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

Finish a plot with `.savefig("path.png")` (saves the figure) or `.show()`
(opens it interactively — in notebooks this renders inline).

```python
(Data("sales.csv")
    .dropna()
    .groupby("city").agg("mean", "salary")
    .plot("bar", x="city", y="salary", title="Mean salary by city")
    .savefig("salary_by_city.png"))
```

Extra matplotlib keyword arguments pass straight through:

```python
d.plot("scatter", x="age", y="salary", color="red", alpha=0.5)
```

> Note: because `dclean` sets a headless-safe matplotlib backend, `savefig`
> always works (e.g. in scripts, CI, servers). `show()` is for interactive use.

---

## Exporting

```python
d.to_csv("cleaned.csv")          # write the current frame, no index
raw = d.to_df()                  # get the raw pandas DataFrame back
raw.describe()                   # now use any pandas method you like
```

`.to_df()` is the escape hatch: `dclean` never hides pandas from you. Use it
for anything `dclean` doesn't wrap yet.

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
d.log()           d.steps()          # what this pipeline actually did
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
d.savefig("chart.png")               # always works (headless-safe)
d.show()                             # interactive / inline in notebooks

d.to_csv("clean.csv")
d.to_df()                            # the raw DataFrame - full pandas
```

---

## Full API reference

| Task | Method | Notes |
|------|--------|-------|
| **Do it all** | `dclean.clean(src, to=...)` `dclean.report(src)` `.clean([nulls])` `.report()` | one call: name the file, the library handles the rest |
| Load file | `Data("file.csv")` | auto-detects csv/xls/xlsx/json/parquet |
| From frame | `Data(df)` / `Data.from_records([...])` | |
| Inspect | `.head(n)` `.tail(n)` `.print([n])` `.to_table([max_rows])` `.shape()` `.dtypes()` `.cols()` `.info()` `.describe()` `.nulls([plot])` | `print(d)` renders the dataset; `repr(d)` stays terse. `.shape()` and `.dtypes()` show per-feature types |
| Clean | `.clean([nulls])` `.dropna([subset])` `.fillna(v)` `.fix_nulls([strategy])` `.drop_outliers([cols])` `.dedupe([subset])` `.drop(c)` `.keep(*c)` `.rename(a=b)` `.lower_cols()` `.astype(a="t")` `.to_float(*cols)` | `.to_float()` strips currency/separators, unparseable→NaN |
| Filter | `.filter("expr")` | `== != > < >= <= and or in not in` + `between` |
| Transform | `.mutate(x="expr")` `.select(*c)` `.sort(by, [ascending])` | |
| Aggregate | `.mean/.sum/.count/.median/.min/.max([col], [by])` `.top(n, by)` `.bottom(n, by)` `.counts(col)` `.stat(how, col, by)` | one call, no separate `groupby` step |
| Aggregate (longhand) | `.groupby(*c).agg(how, col)` `.summarize(**stats)` `.corr([method])` | |
| Plot | `.plot(kind, [x], [y], [title])` `.plot_corr([title])` | line/bar/hist/scatter/box/pie; `x`/`y` inferred on a 2-column frame |
| Output | `.savefig(path)` `.show()` `.to_csv(path)` `.to_df()` | |
| Provenance | `.log()` `.steps()` | what the pipeline actually did |

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
- **Headless-safe plotting.** The matplotlib backend is set to `Agg`, so
  `savefig()` works in scripts, CI, and servers without a display.
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
```

---

## License

MIT — see [LICENSE](LICENSE).
