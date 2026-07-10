# dcleaner

**Fluent, low-boilerplate data cleaning and visualization — built on top of pandas.**

Tired of writing the same `df.dropna().query(...).groupby(...).plot()` boilerplate
every time you explore a dataset? `dcleaner` wraps pandas in a chainable, readable
API so a full **load → clean → filter → aggregate → plot** pipeline is one
expression. It is a thin, fluent layer over pandas — when you need the full
power of pandas, call `.to_df()` and drop straight back into it.

![CI](https://github.com/Shivmahlan/dcleaner/actions/workflows/ci.yml/badge.svg)

---

## Table of contents

- [Install](#install)
- [The idea](#the-idea)
- [Quick start](#quick-start)
- [Loading data](#loading-data)
- [Inspecting](#inspecting)
- [Cleaning](#cleaning)
- [Filtering](#filtering)
- [Transforming](#transforming)
- [Aggregating](#aggregating)
- [Correlations](#correlations)
- [Plotting](#plotting)
- [Exporting](#exporting)
- [Full API reference](#full-api-reference)
- [Why not just use pandas?](#why-not-just-use-pandas)
- [Design notes](#design-notes)
- [License](#license)

---

## Install

```bash
pip install dcleaner
```

Requirements: Python ≥ 3.8, plus `pandas`, `matplotlib`, and `numpy`
(automatically installed as dependencies).

## Try it

A self-contained demo notebook lives at [`examples/demo.ipynb`](examples/demo.ipynb) —
it generates its own data and walks through clean → filter → aggregate → plot.

---

## The idea

Most EDA follows the same shape:

1. Load a file.
2. Drop missing rows, fix types, rename ugly columns.
3. Filter to the rows you care about.
4. Summarize / group / correlate.
5. Plot something to check your intuition.

With raw pandas that's a pile of intermediate variables and method calls.
With `dclean`, every step returns the object itself, so you chain:

```python
from dclean import Data

result = (Data("sales.csv")
    .dropna()
    .filter("age > 18 and city in ['NY', 'LA']")
    .groupby("city").agg("mean", "salary")
    .plot("bar", x="city", y="salary", title="Mean salary by city")
    .savefig("salary_by_city.png"))
```

That single chain replaces ~30 lines of pandas + matplotlib setup.

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
d.shape()           # prints "N rows x M cols"
d.cols()            # prints the column list
d.info()            # pandas .info()
d.describe()        # summary stats of numeric columns
print(d)            # Data(shape=(N, M))  — or just .show() to print the frame
```

---

## Cleaning

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
    .astype(price="float"))   # cast types
```

> Tip: start every pipeline with `.lower_cols()` so you never have to remember
> whether a column is `Price`, `price`, or `PRICE`.

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

Columns whose names contain spaces must be quoted inside the string:

```python
d.filter("'total sales' > 100")
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

# Pass a literal (non-string) value to assign it directly:
d.mutate(flag=True)
```

Other transforms:

```python
d.select("name", "price")   # alias for keep()
d.keep("name", "price")
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

## Full API reference

| Task | Method | Notes |
|------|--------|-------|
| Load file | `Data("file.csv")` | auto-detects csv/xls/xlsx/json/parquet |
| From frame | `Data(df)` / `Data.from_records([...])` | |
| Inspect | `.head(n)` `.tail(n)` `.shape()` `.cols()` `.info()` `.describe()` | print helpers |
| Clean | `.dropna([subset])` `.fillna(v)` `.dedupe([subset])` `.drop(c)` `.keep(c)` `.rename(a=b)` `.lower_cols()` `.astype(a="t")` | |
| Filter | `.filter("expr")` | `== != > < >= <= and or in not in` |
| Transform | `.mutate(x="expr")` `.select(*c)` `.sort(by, [ascending])` | |
| Aggregate | `.groupby(*c).agg(how, col)` `.summarize(**stats)` `.corr([method])` | |
| Plot | `.plot(kind, x, y, [title])` `.plot_corr([title])` | line/bar/hist/scatter/box/pie |
| Output | `.savefig(path)` `.show()` `.to_csv(path)` `.to_df()` | |

All transform/clean/aggregate methods return `self`, so they chain. Inspect
and output methods also return `self` unless they hand back a value
(`.to_df()`, `len(d)`, `repr(d)`).

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

- **Fluent by default.** Every operation returns the object, enabling chains.
- **Vectorized.** `filter()` and `mutate()` use `DataFrame.eval`/`query`, so
  they stay fast on large frames — no Python-row loops.
- **Headless-safe plotting.** The matplotlib backend is set to `Agg`, so
  `savefig()` works in scripts, CI, and servers without a display.
- **Escape hatch.** `.to_df()` gives you the raw DataFrame for anything not
  wrapped.

---

## License

MIT — see [LICENSE](LICENSE).
