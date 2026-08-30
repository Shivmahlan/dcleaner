"""Core fluent DataFrame wrapper for dclean."""
import datetime
import functools
import glob
import importlib
import json
import os
import re
import sys
from html import escape as _escape
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Show every column (no "..." truncation) on any raw pandas print.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)

try:  # tabulate gives clean, non-truncated tables; fall back gracefully
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    def tabulate(df, **kwargs):
        showindex = kwargs.get("showindex", True)
        return df.to_string(index=bool(showindex))

# ----------------------------------------------------------------------- COLOR
# Escape codes are for a terminal that can render them. Redirected into a file,
# a pager or a CI log they are noise wrapped around every heading, so resolve
# them once at import: FORCE_COLOR wins, then NO_COLOR (no-color.org), then the
# honest question - is stdout actually a terminal?
def _color_enabled(stream=None):
    """True when it is safe to emit ANSI escapes on ``stream``."""
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        return force != "0"
    if os.environ.get("NO_COLOR"):
        return False
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except Exception:  # pragma: no cover - a stream with no isatty()
        return False


def _resolve_colors(stream=None):
    """``(BOLD, UNDER, RESET)`` for this environment - all empty when plain."""
    if _color_enabled(stream):
        return "\033[1m", "\033[4m", "\033[0m"
    return "", "", ""


BOLD, UNDER, RESET = _resolve_colors()

# Values that mean "missing" in real-world exports but read as text.
NA_TOKENS = {"", "na", "n/a", "n.a.", "nan", "null", "none", "-", "--", "?", "unknown"}

_DATEISH = re.compile(
    r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}"      # 2025-01-31 / 2025/1/31
    r"|^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"   # 31-01-2025 / 1/31/25
)


# --------------------------------------------------------------------- DISPLAY
# A plot should appear where you are working - inline in a notebook, in a window
# from a script or REPL - not as a PNG you have to go and open. dclean therefore
# pins no backend: it only fills in the one matplotlib cannot guess for itself
# (a Jupyter kernel) and leaves matplotlib's own detection - which already falls
# back to Agg when there is no display - alone everywhere else.
_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
_NOTEBOOK_BACKENDS = ("inline", "ipympl", "nbagg", "widget")


def _in_jupyter():
    """True inside a Jupyter/IPython kernel (not a plain terminal IPython)."""
    try:
        shell = sys.modules["IPython"].get_ipython()
    except (KeyError, AttributeError):  # pragma: no cover - IPython not installed
        return False
    return shell is not None and hasattr(shell, "kernel")


def _backend_unset():
    """True when nobody has told matplotlib which backend to use."""
    try:
        from matplotlib import rcsetup
        return (dict.__getitem__(matplotlib.rcParams, "backend")
                is rcsetup._auto_backend_sentinel)
    except Exception:  # pragma: no cover - private API moved; assume "chosen"
        return False


def _autoselect_backend():
    """Pick the inline backend in notebooks; never override an explicit choice."""
    if os.environ.get("MPLBACKEND") or not _backend_unset():
        return  # the user (or a test) asked for a specific backend - respect it
    if _in_jupyter():
        try:
            matplotlib.use("module://matplotlib_inline.backend_inline")
        except Exception:  # pragma: no cover - matplotlib_inline not installed
            pass


_autoselect_backend()


def _display_mode():
    """Where a figure can be shown: ``notebook``, ``gui`` or ``headless``."""
    backend = matplotlib.get_backend().lower()
    if any(k in backend for k in _NOTEBOOK_BACKENDS):
        return "notebook"
    if backend in _NON_INTERACTIVE_BACKENDS:
        return "headless"
    return "gui"


def _render(fig, show):
    """Put ``fig`` in front of the user, or hold it for later.

    ``show=None`` means auto: render inline in a notebook, where it is free and
    is exactly what you expect; stay quiet in a script, where popping a blocking
    window mid-pipeline would hijack the run - call ``.show()`` there.
    """
    mode = _display_mode()
    if show is None:
        show = mode == "notebook"
    if show and mode == "notebook":
        try:
            from IPython.display import display
            display(fig)
            plt.close(fig)  # shown already; stops the duplicate at end of cell
        except ImportError:  # pragma: no cover - inline backend without IPython
            plt.show()
    elif show and mode == "gui":
        plt.show()
    elif show:
        print("no display available (matplotlib backend "
              f"'{matplotlib.get_backend()}') - save the plot instead: "
              ".savefig('plot.png')")
    elif mode == "notebook":
        plt.close(fig)  # show=False means show=False, even in a notebook
    return fig


def _listify(x):
    """One column name or several -> a list of names."""
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _dtype_kind(s):
    """Coarse type of a Series: what has to agree for a join key to work."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    return "text"


def _key_tuples(df, keys):
    """Rows of ``keys`` as hashable tuples - for set membership on any key."""
    return [tuple(row) for row in df[keys].astype(object).to_numpy()]


def _folded_key_tuples(df, keys):
    """Key tuples with case and padding removed - for near-miss diagnosis."""
    folded = df[keys].astype(str).apply(lambda c: c.str.strip().str.lower())
    return [tuple(row) for row in folded.to_numpy()]


def _rows(n):
    """`1 row` / `3 rows` - reports that read like English."""
    return f"{n} row" + ("" if n == 1 else "s")


def _fmt_keys(values, limit=3):
    """`'SF ', 'chicago' (+2 more)` - a readable sample of key values."""
    shown = [repr(v[0]) if len(v) == 1 else repr(tuple(v)) for v in values[:limit]]
    more = len(values) - len(shown)
    return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


class _classorinstance:
    """A method that works both as ``Data.concat(...)`` and ``d.concat(...)``.

    A plain ``classmethod`` called on an instance silently discards that
    instance, so ``d.concat(other)`` would drop ``d``'s own rows - the wrong
    answer, quietly. This passes the instance through (``None`` off the class).
    """

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, obj, objtype=None):
        return functools.partial(self.func, obj)


def _clean_numeric_strings(s):
    """Strip currency symbols, thousands separators and stray spaces from a Series."""
    return (s.astype(str)
             .str.strip()
             .str.replace(r"[,\s]", "", regex=True)
             .str.replace(r"^[\$€£¥₹]", "", regex=True)
             .str.replace(r"%$", "", regex=True)
             .str.replace(r"^\((.*)\)$", r"-\1", regex=True))  # (123) -> -123


def _reject_dunder(expr):
    """Refuse expressions reaching for dunder attributes.

    ``filter()`` and ``mutate()`` execute their argument, so the string must
    come from the developer, never from an end user. This blocks the usual
    escape route out of a restricted namespace - it is a guard rail, not a
    sandbox, and it is not a licence to pass untrusted input.
    """
    if "__" in str(expr):
        raise ValueError(
            "expression contains '__', which dclean refuses to evaluate. "
            "filter()/mutate() strings are executed as code and must be "
            "written by you, never taken from user input.")
    return expr


# ------------------------------------------------------------------ HTML REPORT
# A profile is worth sharing - with a reviewer, a ticket, a data owner who does
# not have the file. That means ONE file that opens anywhere: styles inline, no
# CDN link, no fonts to fetch, nothing that breaks behind a firewall or in an
# email attachment.
_REPORT_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       Helvetica, Arial, sans-serif; line-height: 1.5; margin: 0;
       padding: 2rem 1.25rem; background: #fbfbfa; color: #1a1a1a; }
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 0.25rem; }
h2 { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.06em;
     color: #6b6b6b; margin: 2rem 0 0.6rem; }
.sub { color: #6b6b6b; margin: 0 0 0.5rem; font-size: 0.95rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 0.88rem; width: 100%;
        font-variant-numeric: tabular-nums; }
th, td { padding: 0.4rem 0.7rem; text-align: left; white-space: nowrap;
         border-bottom: 1px solid #e6e6e3; }
thead th { background: #f0f0ed; font-weight: 600; }
tbody tr:hover td { background: #f6f6f4; }
ul.warn { list-style: none; padding: 0; margin: 0; }
ul.warn li { padding: 0.5rem 0.75rem; margin-bottom: 0.35rem; border-radius: 4px;
             background: #fff6e5; border-left: 3px solid #d98b00; }
p.ok { padding: 0.5rem 0.75rem; border-radius: 4px; background: #eaf6ec;
       border-left: 3px solid #2e7d43; margin: 0; }
.note { font-size: 0.85rem; color: #6b6b6b; margin-top: 0.4rem; }
footer { margin-top: 2.5rem; font-size: 0.8rem; color: #8a8a8a; }
@media (prefers-color-scheme: dark) {
  body { background: #17171a; color: #e8e8e6; }
  h2, .sub, .note, footer { color: #9a9a97; }
  th, td { border-bottom-color: #2e2e33; }
  thead th { background: #232327; }
  tbody tr:hover td { background: #1e1e22; }
  ul.warn li { background: #2c2413; border-left-color: #d98b00; }
  p.ok { background: #14251a; border-left-color: #2e7d43; }
}
"""


def _html_report(profile, title="dclean profile"):
    """Render a ``_profile()`` dict as ONE self-contained HTML document.

    Every value that came from the data is escaped, and every style is inline:
    the file opens with no network and nothing else beside it.
    """
    rows, cols = profile["rows"], profile["cols"]
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(title)}</title>",
        f"<style>{_REPORT_CSS}</style>",
        "</head><body><main>",
        f"<h1>{_escape(title)}</h1>",
        f'<p class="sub">{rows} rows &times; {cols} cols '
        f'&middot; {profile["kb"]:.1f} KB in memory '
        f'&middot; {profile["dupes"]} duplicate rows</p>',
        "<h2>Columns</h2>",
        '<div class="scroll">'
        + pd.DataFrame(profile["table"]).to_html(index=False, border=0)
        + "</div>",
    ]
    if not profile["examples"]:
        parts.append('<p class="note">Example values are withheld '
                     "(<code>examples=False</code>) - the type is shown "
                     "instead, so this file carries no data of yours.</p>")
    num = profile["numeric"]
    if num is not None:
        parts += ["<h2>Numeric summary</h2>",
                  '<div class="scroll">'
                  + num.to_html(border=0, float_format=lambda v: f"{v:.2f}")
                  + "</div>"]
    parts.append("<h2>Warnings</h2>")
    if profile["warnings"]:
        parts.append('<ul class="warn">'
                     + "".join(f"<li>{_escape(w)}</li>" for w in profile["warnings"])
                     + "</ul>")
    else:
        parts.append('<p class="ok">No data-quality warnings.</p>')
    parts += ["<footer>Generated by dcleaner &middot; "
              "<code>d.report(to=&quot;profile.html&quot;)</code></footer>",
              "</main></body></html>"]
    return "\n".join(parts)


# --------------------------------------------------------------------- RECIPES
# A pipeline is worth replaying: the same cleaning, next month's file. The log
# already knows what happened, but it knows it as display strings - and
# "drop(['a', 'b'])" cannot be turned back into a call without a parser for
# dclean's own repr, which would be wrong the first time a column name contains
# a quote. So each step carries a STRUCTURED record beside its display string,
# and replay reads the structure and ignores the prose.
RECIPE_FORMAT = 1

# Replayable: a pure function of the frame plus JSON-serializable arguments.
# join()/concat() need a second table a recipe cannot carry; groupby().agg() is
# two calls - .mean(col, by=...) is the one-call form that records.
REPLAYABLE = frozenset((
    "clean", "fix_nulls", "drop_outliers", "dropna", "fillna", "drop", "keep",
    "rename", "dedupe", "astype", "lower_cols", "to_float", "filter", "mutate",
    "select", "sort", "copy",
    "stat", "mean", "sum", "count", "median", "min", "max",
    "top", "bottom", "counts", "summarize", "corr",
))

_NOT_REPLAYABLE_HINT = {
    "join": "join() needs the other table, which a recipe cannot carry",
    "concat": "concat() needs the other sources, which a recipe cannot carry",
    "groupby": "groupby().agg() is two calls - use .mean(col, by=...) "
               "(or .sum/.count/...), which records as one",
    "agg": "groupby().agg() is two calls - use .mean(col, by=...) "
           "(or .sum/.count/...), which records as one",
}


def _step(display, method=None, args=None, kwargs=None):
    """One entry in the pipeline log.

    ``display`` is what a human reads; ``method``/``args``/``kwargs`` are what
    a recipe replays. A step with no ``method`` is recorded but cannot replay.
    """
    return {
        "display": display,
        "method": method,
        "args": list(args) if args else [],
        "kwargs": dict(kwargs) if kwargs else {},
    }


def _as_step(step):
    """Accept a step dict or a bare display string (the older shape)."""
    return dict(step) if isinstance(step, dict) else _step(str(step))


def _jsonable(value):
    """True when ``value`` survives a JSON round trip unchanged."""
    try:
        return json.loads(json.dumps(value)) == value
    except (TypeError, ValueError):
        return False


def _version():
    """The installed dcleaner version, recorded in a recipe for provenance."""
    try:
        from . import __version__
        return __version__
    except Exception:  # pragma: no cover - partially initialised package
        return "unknown"


def _require(modules, what, install=None):
    """Import the first available of ``modules``, or say what to pip install.

    Optional dependencies are imported at the point of use, never at import
    time, so ``pip install dcleaner`` stays small - and the error names the
    exact command that fixes it instead of a bare ImportError.
    """
    names = [modules] if isinstance(modules, str) else list(modules)
    for name in names:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError(
        "{} needs {}, which dclean does not install for you. "
        "Run:  pip install {}".format(
            what, " or ".join(repr(n) for n in names), install or names[0]))


def _normalize_name(name):
    """`  Total Sales ($) ` -> `total_sales`."""
    n = str(name).strip().lower()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"[\s_]+", "_", n).strip("_")
    return n or "column"


class Data:
    """Fluent, immutable wrapper around a pandas DataFrame.

    Every transform returns a NEW ``Data`` - the object you called it on is
    never modified - so calls chain safely and intermediate variables keep
    the value they were assigned:

        from dclean import Data
        raw = Data("sales.csv")
        clean = raw.dropna()      # `raw` still holds every original row

        (Data("sales.csv")
         .dropna()
         .filter("age > 18")
         .groupby("city").agg("mean", "salary")
         .plot("bar", x="city", y="salary")
         .show())

    Inspect methods (``head``, ``nulls``, ``report`` ...) print and return the
    same object, since they change nothing. Drop back to raw pandas anytime
    with ``.to_df()``.
    """

    def __init__(self, source=None, df=None, steps=None):
        if df is not None:
            self.df = df.copy()
        elif isinstance(source, pd.DataFrame):
            self.df = source.copy()
        elif isinstance(source, str):
            self.df = self._load(source)
        elif source is None:
            self.df = pd.DataFrame()
        else:
            raise TypeError(f"Data() can't handle source of type {type(source).__name__}")
        self._group = None
        self._fig = None
        self._steps = [_as_step(s) for s in steps] if steps else []

    # ----------------------------------------------------------- INTERNAL
    def _derive(self, df, step=None, group=None, call=None):
        """Build the next Data in the chain. Never touches ``self``.

        ``df`` is adopted as-is (pandas ops already return new frames), so a
        chain costs no redundant copies.

        ``step`` is the line a human reads in ``log()``. ``call`` is the
        ``(method, args, kwargs)`` a recipe replays - passed by the transform,
        which already knows its own arguments, instead of being parsed back out
        of the display string later. A transform with no ``call`` still logs;
        it just cannot be replayed.
        """
        out = object.__new__(Data)
        out.df = df
        out._group = group
        out._fig = self._fig
        entry = None
        if step is not None:
            method, args, kwargs = call if call else (None, (), {})
            entry = _step(step, method, args, kwargs)
        out._steps = self._steps + ([entry] if entry else [])
        return out

    # ----------------------------------------------------------- LOAD
    @staticmethod
    def _sample_path(name):
        """Resolve a bare filename against the bundled sample datasets."""
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "data", name)
        return cand if os.path.exists(cand) else None

    @staticmethod
    def _load(path):
        # Bare name (no directory) that isn't on disk -> try bundled samples,
        # so `Data("sample_sales.csv")` works after a plain `pip install`.
        if not os.path.exists(path) and not os.path.dirname(str(path)):
            bundled = Data._sample_path(path)
            if bundled:
                path = bundled
        if path.endswith(".csv") or path.endswith(".csv.gz"):
            return pd.read_csv(path)
        if path.endswith((".xls", ".xlsx")):
            return pd.read_excel(path)
        if path.endswith(".json"):
            return pd.read_json(path)
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file type: {path}")

    @classmethod
    def from_records(cls, records):
        """Build from a list of dicts."""
        return cls(df=pd.DataFrame(records))

    @staticmethod
    def samples():
        """List the dataset names bundled with the package."""
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        names = sorted(f for f in os.listdir(here) if f.endswith(".csv"))
        print("bundled datasets:", ", ".join(names) if names else "(none)")
        return names

    # ----------------------------------------------------------- INSPECT
    def head(self, n=5):
        print(tabulate(self.df.head(n), headers="keys", tablefmt="github",
                       showindex=False))
        return self

    def tail(self, n=5):
        print(tabulate(self.df.tail(n), headers="keys", tablefmt="github",
                       showindex=False))
        return self

    def to_table(self, max_rows=None):
        """Render the FULL dataset as a formatted table (no column truncation).

        Pass ``max_rows`` to cap the number of printed rows - every column is
        always shown. Useful when ``head()`` / ``tail()`` only show part of the
        data and you want the whole thing on screen.
        """
        df = self.df if max_rows is None else self.df.head(max_rows)
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
        return self

    def print(self, n=None):
        """Print the dataset itself (chainable).

        No argument -> print the FULL frame. Pass ``n`` to cap to the first
        ``n`` rows (like head()). Returns the same object so it can sit inside
        a pipeline, e.g. dump the cleaned data right before exporting it::

            (Data("sales.csv")
             .dropna()
             .print()                 # dump the cleaned data to stdout
             .filter("age > 18")
             .to_csv("out.csv"))

        For a quick look at just the head, use ``.head()`` instead.
        """
        df = self.df if n is None else self.df.head(n)
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
        return self

    def info(self):
        self.df.info()
        return self

    def shape(self):
        """Print the shape, the column list, and each feature's dtype."""
        print(f"{self.df.shape[0]} rows x {self.df.shape[1]} cols")
        print("columns:", list(self.df.columns))
        print("dtypes:")
        print(self.df.dtypes.to_string())
        return self

    def dtypes(self):
        """Print each feature's data type (a tidy ``column -> dtype`` list).

        For just the raw pandas Series, use ``.to_df().dtypes``.
        """
        print(self.df.dtypes.to_string())
        return self

    def cols(self):
        print(list(self.df.columns))
        return self

    def describe(self):
        """Pretty, highlighted summary of numeric columns.

        Prints a banner + a tidy table, then calls out the headline statistic
        (the **mean**) per column so the main description jumps out at a glance.
        Frames with no numeric columns fall back to pandas' object summary.
        """
        if self.df.empty:
            print("(no data to describe)")
            return self
        num = self.df.select_dtypes(include="number")
        if num.empty:
            print(f"\n{BOLD}{UNDER}SUMMARY (no numeric columns){RESET}")
            print(tabulate(self.df.describe(), headers="keys",
                           tablefmt="github", showindex=True))
            return self
        desc = num.describe()
        print(f"\n{BOLD}{UNDER}DESCRIPTIVE STATISTICS (numeric columns){RESET}")
        print(tabulate(desc, headers="keys", tablefmt="github",
                       showindex=True, floatfmt=".3f"))
        means = desc.loc["mean"].to_dict()
        if means:
            callout = ", ".join(f"{k}: {v:.3f}" for k, v in means.items())
            print(f"{BOLD}-> mean | {callout}{RESET}\n")
        return self

    def nulls(self, plot=False, show=None):
        """Show missing-value counts per column (and the total).

        Returns the same object so it can sit in a chain right before
        ``.dropna()``. Set ``plot=True`` to also chart the null counts - it
        renders inline in a notebook; from a script add ``.show()`` for a
        window or ``.savefig()`` for a file.
        """
        counts = self.df.isna().sum()
        total = int(counts.sum())
        report = counts.rename("nulls").reset_index()
        report.columns = ["column", "nulls"]
        rows = max(len(self.df), 1)
        report["pct"] = (report["nulls"] / rows * 100).round(1)
        print(tabulate(report, headers="keys", tablefmt="github",
                       showindex=False))
        print(f"{BOLD}-> total nulls: {total} across {len(self.df)} rows{RESET}")
        if plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_title("Missing values per column")
            ax.set_ylabel("nulls")
            plt.xticks(rotation=45, ha="right")
            self._fig = _render(fig, show)
        return self

    def _profile(self, examples=True):
        """Measure the dataset once; ``report()`` decides how to show it.

        Internal. Returns a plain dict so the terminal and the HTML file are
        rendered from exactly the same numbers - they can never drift.
        """
        df = self.df
        rows = max(len(df), 1)
        table = []
        for c in df.columns:
            col = df[c]
            n_null = int(col.isna().sum())
            sample = col.dropna()
            if sample.empty:
                shown = ""
            elif examples:
                shown = str(sample.iloc[0])[:24]
            else:
                shown = f"<{type(sample.iloc[0]).__name__}>"
            table.append({
                "column": c,
                "dtype": str(col.dtype),
                "nulls": n_null,
                "null %": round(n_null / rows * 100, 1),
                "unique": int(col.nunique(dropna=True)),
                "example": shown,
            })
        num = df.select_dtypes(include="number")
        return {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "kb": df.memory_usage(deep=True).sum() / 1024,
            "table": table,
            "dupes": int(df.duplicated().sum()),
            "numeric": None if num.empty else num.describe(),
            "warnings": self._warnings(),
            "examples": bool(examples),
        }

    def report(self, examples=True, to=None):
        """One-call profile of the whole dataset.

        Inspector: shows the profile and returns the SAME object.

        Reports shape, a per-column breakdown (dtype, nulls, unique values, a
        sample value), duplicate-row count, numeric summary stats, and a list
        of data-quality warnings. This is the "what am I even looking at"
        button - run it the moment you load a file.

        The ``example`` column shows a real value from your data. Pass
        ``examples=False`` to replace it with the value's type, which is what
        you want before pasting a profile into a ticket, a log, or CI output
        on data containing anything personal.

        ``to="profile.html"`` writes the same profile to a self-contained HTML
        file instead of printing it - one file, styles inline, nothing fetched
        from a CDN, so it survives being emailed or attached to a ticket.
        ``examples=False`` is honoured there too, and matters more: that file
        is the one somebody else ends up holding.
        """
        profile = self._profile(examples)
        if to:
            title = f"dclean profile - {os.path.basename(str(to))}"
            with open(to, "w", encoding="utf-8") as fh:
                fh.write(_html_report(profile, title))
            print(f"saved report -> {to}")
            return self

        print(f"\n{BOLD}{UNDER}DATASET REPORT{RESET}")
        print(f"{profile['rows']} rows x {profile['cols']} cols  "
              f"| {profile['kb']:.1f} KB in memory")
        print(tabulate(pd.DataFrame(profile["table"]), headers="keys",
                       tablefmt="github", showindex=False))
        print(f"duplicate rows: {profile['dupes']}")

        if profile["numeric"] is not None:
            print(f"\n{BOLD}numeric summary{RESET}")
            print(tabulate(profile["numeric"], headers="keys",
                           tablefmt="github", showindex=True, floatfmt=".2f"))

        if profile["warnings"]:
            print(f"\n{BOLD}warnings{RESET}")
            for w in profile["warnings"]:
                print(f"  ! {w}")
        else:
            print(f"\n{BOLD}-> no data-quality warnings{RESET}")
        print()
        return self

    def _warnings(self):
        """Data-quality issues worth a human's attention."""
        df, out = self.df, []
        rows = max(len(df), 1)
        if df.empty:
            return ["dataset is empty"]
        if int(df.duplicated().sum()):
            out.append(f"{int(df.duplicated().sum())} duplicate rows "
                       f"- .clean() or .dedupe() removes them")
        for c in df.columns:
            col = df[c]
            null_pct = col.isna().sum() / rows * 100
            if null_pct == 100:
                out.append(f"'{c}' is entirely empty")
            elif null_pct > 40:
                out.append(f"'{c}' is {null_pct:.0f}% missing")
            if col.nunique(dropna=True) <= 1 and null_pct < 100:
                out.append(f"'{c}' is constant - carries no signal")
            if col.dtype == object:
                stripped = _clean_numeric_strings(col.dropna())
                if len(stripped) and pd.to_numeric(
                        stripped, errors="coerce").notna().mean() > 0.9:
                    out.append(f"'{c}' looks numeric but is stored as text "
                               f"- .clean() or .to_float('{c}') fixes it")
                elif col.nunique(dropna=True) == len(col.dropna()) and len(col.dropna()) > 1:
                    out.append(f"'{c}' is unique per row - looks like an ID")
        return out

    def log(self):
        """Print the steps that produced this dataset."""
        if not self._steps:
            print("(no transforms applied yet)")
        else:
            print(f"{BOLD}pipeline{RESET}")
            for i, s in enumerate(self._steps, 1):
                print(f"  {i}. {s['display']}")
        return self

    def steps(self):
        """Return the applied-step log as a list of strings.

        Escape hatch: returns plain values. For the structured form a recipe
        replays, use :meth:`save_recipe`.
        """
        return [s["display"] for s in self._steps]

    # ----------------------------------------------------------- AUTO
    def clean(self, nulls="keep", dates=True, verbose=True):
        """Auto-clean the dataset in one call.

        You say *clean it*; this works out the rest. It:

        1. normalizes column names (``" Total Sales "`` -> ``total_sales``),
        2. drops all-empty columns and all-empty rows,
        3. trims whitespace on text columns and turns ``""``/``"n/a"``/``"null"``
           into real NaN,
        4. converts text columns that are really numbers to numbers - money and
           thousands separators included (``"$1,234.50"`` -> ``1234.5``),
        5. parses date-looking text columns into datetimes (``dates=False``
           to skip),
        6. removes duplicate rows,
        7. handles missing values per ``nulls``.

        ``nulls`` - ``"keep"`` (default, leave NaN in place), ``"drop"`` (drop
        rows with any NaN), or ``"fill"`` (median for numeric columns, mode for
        the rest).

        Set ``verbose=False`` to clean silently. Returns a NEW ``Data``.
        """
        if nulls not in ("keep", "drop", "fill"):
            raise ValueError("nulls must be 'keep', 'drop' or 'fill'")
        df = self.df.copy()
        before_rows, before_cols = df.shape
        actions = []

        # 1. column names
        renames = {c: _normalize_name(c) for c in df.columns}
        if any(k != v for k, v in renames.items()):
            df = df.rename(columns=renames)
            df = self._dedupe_column_names(df)
            actions.append(f"normalized {sum(k != v for k, v in renames.items())} column names")

        # 2/3. blank-out NA tokens, trim text, then drop empty rows/cols
        text_cols = list(df.select_dtypes(include="object").columns)
        for c in text_cols:
            s = df[c].astype(str).str.strip()
            df[c] = s.mask(s.str.lower().isin(NA_TOKENS))
        if text_cols:
            actions.append(f"trimmed whitespace on {len(text_cols)} text columns")

        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            actions.append(f"dropped {len(empty_cols)} empty columns: {', '.join(map(str, empty_cols))}")
        n_before = len(df)
        df = df.dropna(how="all")
        if len(df) != n_before:
            actions.append(f"dropped {n_before - len(df)} empty rows")

        # 4. numeric-looking text -> numbers
        # A column converts when >90% of its values parse, so up to 10% of real
        # values can be destroyed. Count them and say so - silent loss is not
        # acceptable on data anyone cares about.
        converted, lost = [], {}
        for c in df.select_dtypes(include="object").columns:
            vals = df[c].dropna()
            if vals.empty:
                continue
            num = pd.to_numeric(_clean_numeric_strings(vals), errors="coerce")
            if num.notna().mean() > 0.9:
                casualties = int(num.isna().sum())
                df[c] = pd.to_numeric(_clean_numeric_strings(df[c]), errors="coerce")
                converted.append(str(c))
                if casualties:
                    lost[str(c)] = casualties
        if converted:
            actions.append(f"converted to numeric: {', '.join(converted)}")
        for c, n in lost.items():
            actions.append(f"WARNING {n} value(s) in '{c}' could not be parsed "
                           f"as numbers and became NaN")

        # 5. date-looking text -> datetime
        parsed = []
        if dates:
            for c in df.select_dtypes(include="object").columns:
                vals = df[c].dropna().astype(str)
                if vals.empty or vals.str.match(_DATEISH).mean() <= 0.9:
                    continue
                out = pd.to_datetime(df[c], errors="coerce")
                if out.notna().sum() >= df[c].notna().sum() * 0.9:
                    casualties = int(df[c].notna().sum() - out.notna().sum())
                    df[c] = out
                    parsed.append(str(c))
                    if casualties:
                        actions.append(f"WARNING {casualties} value(s) in '{c}' "
                                       f"could not be parsed as dates and became NaT")
        if parsed:
            actions.append(f"parsed as dates: {', '.join(parsed)}")

        # 6. duplicates
        n_dupes = int(df.duplicated().sum())
        if n_dupes:
            df = df.drop_duplicates()
            actions.append(f"removed {n_dupes} duplicate rows")

        # 7. missing values
        if nulls == "drop":
            n_before = len(df)
            df = df.dropna()
            actions.append(f"dropped {n_before - len(df)} rows containing nulls")
        elif nulls == "fill":
            filled = []
            for c in df.columns:
                if not df[c].isna().any():
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
                else:
                    mode = df[c].mode(dropna=True)
                    if not mode.empty:
                        df[c] = df[c].fillna(mode.iloc[0])
                filled.append(str(c))
            if filled:
                actions.append(f"filled nulls in: {', '.join(filled)}")

        df = df.reset_index(drop=True)
        if verbose:
            print(f"\n{BOLD}{UNDER}CLEAN{RESET}")
            if actions:
                for a in actions:
                    warn = a.startswith("WARNING ")
                    print(f"  {'!' if warn else '+'} {a[8:] if warn else a}")
            else:
                print("  (already clean - nothing to do)")
            print(f"{BOLD}-> {before_rows}x{before_cols} to "
                  f"{df.shape[0]}x{df.shape[1]}, "
                  f"{int(df.isna().sum().sum())} nulls remaining{RESET}\n")
        return self._derive(df, step=f"clean(nulls={nulls!r})",
                            call=("clean", (), {"nulls": nulls, "dates": bool(dates)}))

    @staticmethod
    def _dedupe_column_names(df):
        """After normalization two columns can collide - suffix the repeats."""
        seen, cols = {}, []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        df.columns = cols
        return df

    def fix_nulls(self, strategy="auto", subset=None):
        """Fill missing values without you picking a statistic per column.

        ``strategy="auto"`` (default) uses the median for numeric columns and
        the most common value for everything else. Pass ``"mean"``, ``"median"``,
        ``"mode"``, ``"zero"``, or ``"ffill"`` to force one. ``subset`` limits
        it to named columns. Returns a NEW ``Data``.
        """
        valid = ("auto", "mean", "median", "mode", "zero", "ffill")
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}, got {strategy!r}")
        df = self.df.copy()
        cols = list(df.columns) if subset is None else (
            [subset] if isinstance(subset, str) else list(subset))
        for c in cols:
            if not df[c].isna().any():
                continue
            numeric = pd.api.types.is_numeric_dtype(df[c])
            if strategy == "ffill":
                df[c] = df[c].ffill().bfill()
            elif strategy == "zero":
                df[c] = df[c].fillna(0)
            elif strategy == "mean" and numeric:
                df[c] = df[c].fillna(df[c].mean())
            elif strategy == "median" and numeric:
                df[c] = df[c].fillna(df[c].median())
            elif strategy in ("mode",) or not numeric:
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    df[c] = df[c].fillna(mode.iloc[0])
            else:  # auto + numeric
                df[c] = df[c].fillna(df[c].median())
        return self._derive(df, step=f"fix_nulls({strategy!r})",
                            call=("fix_nulls", (), {"strategy": strategy,
                                                    "subset": subset}))

    def drop_outliers(self, cols=None, method="iqr", factor=1.5):
        """Drop rows whose numeric values are statistical outliers.

        ``method="iqr"`` (default) removes points outside
        ``Q1 - factor*IQR .. Q3 + factor*IQR``; ``method="zscore"`` removes
        points more than ``factor`` standard deviations from the mean (pass
        ``factor=3`` for the usual rule). Returns a NEW ``Data``.
        """
        df = self.df
        targets = (list(df.select_dtypes(include="number").columns)
                   if cols is None else
                   ([cols] if isinstance(cols, str) else list(cols)))
        mask = pd.Series(True, index=df.index)
        for c in targets:
            s = df[c]
            if method == "zscore":
                sd = s.std()
                if not sd or pd.isna(sd):
                    continue          # no spread -> nothing is an outlier
                keep = (s - s.mean()).abs() <= factor * sd
            else:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                keep = (s >= q1 - factor * iqr) & (s <= q3 + factor * iqr)
            mask &= keep | s.isna()
        removed = int((~mask).sum())
        print(f"{BOLD}-> dropped {removed} outlier rows ({method}){RESET}")
        return self._derive(df[mask].reset_index(drop=True),
                            step=f"drop_outliers({method!r}, factor={factor})",
                            call=("drop_outliers", (), {"cols": cols,
                                                        "method": method,
                                                        "factor": factor}))

    # ----------------------------------------------------------- CLEAN
    def dropna(self, subset=None):
        return self._derive(self.df.dropna(subset=subset), step="dropna()",
                            call=("dropna", (), {"subset": subset}))

    def fillna(self, value):
        return self._derive(self.df.fillna(value), step="fillna()",
                            call=("fillna", (value,), {}))

    def drop(self, cols):
        cols = [cols] if isinstance(cols, str) else list(cols)
        return self._derive(self.df.drop(columns=cols),
                            step=f"drop({cols})",
                            call=("drop", (cols,), {}))

    def keep(self, *cols):
        cols = [c for c in cols if isinstance(c, str)]
        return self._derive(self.df[cols], step=f"keep({cols})",
                            call=("keep", cols, {}))

    def rename(self, **kwargs):
        return self._derive(self.df.rename(columns=kwargs), step="rename()",
                            call=("rename", (), kwargs))

    def dedupe(self, subset=None):
        return self._derive(self.df.drop_duplicates(subset=subset),
                            step="dedupe()",
                            call=("dedupe", (), {"subset": subset}))

    def astype(self, **kwargs):
        return self._derive(self.df.astype(kwargs), step="astype()",
                            call=("astype", (), kwargs))

    def lower_cols(self):
        """Rename all columns to lowercase (common cleaning step)."""
        return self._derive(
            self.df.rename(columns={c: str(c).lower() for c in self.df.columns}),
            step="lower_cols()", call=("lower_cols", (), {}))

    def to_float(self, *cols):
        """Convert string/object column(s) to float.

        Named columns: ``.to_float("price", "qty")``. With no arguments, every
        object/string column is coerced: ``.to_float()``. Currency symbols and
        thousands separators are stripped first, so ``"$1,234.50"`` becomes
        ``1234.5``; anything still unparseable becomes NaN.
        """
        df = self.df.copy()
        targets = list(cols) if cols else list(
            df.select_dtypes(include="object").columns)
        for c in targets:
            df[c] = pd.to_numeric(_clean_numeric_strings(df[c]), errors="coerce")
        return self._derive(df, step=f"to_float({list(targets)})",
                            call=("to_float", tuple(cols), {}))

    # ----------------------------------------------------------- FILTER
    def filter(self, expr):
        """Filter with a readable expression string.

        Supports: == != > < >= <= and or in not in
        e.g. filter("age > 18 and city in ['NY','LA']")
        Also supports a SQL-style `between`: filter("score between 70 and 100")
        resolves to score >= 70 and score <= 100.
        Columns with spaces must use back-ticks: filter("`total sales` > 100")
        """
        return self._derive(self._apply_expr(expr), step=f"filter({expr!r})",
                            call=("filter", (expr,), {}))

    def _apply_expr(self, expr):
        """Return the frame filtered by ``expr`` (mask or query result)."""
        result = self._eval_expr(expr)
        if isinstance(result, pd.DataFrame):
            return result          # query() already returned filtered rows
        return self.df[result]

    def _eval_expr(self, expr):
        _reject_dunder(expr)
        # SQL-style `between x and y` -> `x >= lo and x <= hi`
        m = re.match(r"^\s*(.+?)\s+between\s+(.+?)\s+and\s+(.+?)\s*$", expr, re.IGNORECASE)
        if m:
            col, lo, hi = m.group(1), m.group(2), m.group(3)
            expr = f"({col} >= {lo} and {col} <= {hi})"
        try:
            return self.df.eval(expr)
        except Exception:
            return self.df.query(expr, engine="python")

    # ----------------------------------------------------------- TRANSFORM
    def mutate(self, **kwargs):
        """Add/overwrite columns from expressions.

        mutate(bmi="weight / (height**2)", age1="age + 1")
        A non-string value is assigned literally.
        """
        df = self.df.copy()
        for col, expr in kwargs.items():
            df[col] = self._eval_assign(df, expr) if isinstance(expr, str) else expr
        return self._derive(df, step=f"mutate({list(kwargs)})",
                            call=("mutate", (), kwargs))

    @staticmethod
    def _eval_assign(df, expr):
        """Evaluate a column expression.

        ``DataFrame.eval`` is tried first (fast, vectorized). It cannot parse
        accessor calls like ``price.str.strip()``, so those fall back to a
        plain evaluation over the frame's columns - same trust model as
        ``DataFrame.query``.
        """
        _reject_dunder(expr)
        try:
            return df.eval(expr)
        except Exception:
            ns = {str(c): df[c] for c in df.columns}
            return eval(expr, {"__builtins__": {}}, ns)  # noqa: S307

    def select(self, *cols):
        return self._derive(self.df[list(cols)], step=f"select({list(cols)})",
                            call=("select", cols, {}))

    def sort(self, by, ascending=True):
        return self._derive(self.df.sort_values(by, ascending=ascending),
                            step=f"sort({by!r})",
                            call=("sort", (by,), {"ascending": ascending}))

    # ----------------------------------------------------------- COMBINE
    @staticmethod
    def _as_frame(source):
        """``Data`` / ``DataFrame`` / file path -> a DataFrame."""
        if isinstance(source, Data):
            return source.df
        if isinstance(source, pd.DataFrame):
            return source
        if isinstance(source, str):
            return Data(source).df
        raise TypeError("expected a Data, a DataFrame or a file path, "
                        f"got {type(source).__name__}")

    @staticmethod
    def _label(source, i):
        """What to call a source in the report."""
        return os.path.basename(source) if isinstance(source, str) else f"source {i}"

    def join(self, other, on=None, how="left", left_on=None, right_on=None,
             suffix="_right", verbose=True):
        """Join another table onto this one - and say what actually matched.

            d.join("regions.csv", on="city")
            d.join(other, on=["city", "year"], how="inner")
            d.join(other, left_on="city_id", right_on="id")

        ``other`` is a ``Data``, a DataFrame, or a path. ``on`` defaults to the
        columns the two tables share. ``how`` is ``left`` (default), ``inner``,
        ``right`` or ``outer``.

        The report is the point. A join that silently matches nothing - because
        one side says ``"SF "`` and the other ``"SF"`` - is the classic way to
        get quietly wrong numbers, so this counts the rows that matched, shows
        the keys that did not, tells you when they would match after
        ``.clean()``, and warns when a non-unique key multiplied your rows.
        Set ``verbose=False`` to skip it. Returns a NEW ``Data``.
        """
        if how not in ("left", "right", "inner", "outer"):
            raise ValueError("how must be 'left', 'right', 'inner' or 'outer'")
        right = self._as_frame(other)
        left = self.df

        # ---- work out the key
        inferred = False
        if on is not None:
            lk = rk = _listify(on)
        elif left_on is not None or right_on is not None:
            if left_on is None or right_on is None:
                raise ValueError("left_on and right_on must be given together "
                                 "(or use on= for a shared column name)")
            lk, rk = _listify(left_on), _listify(right_on)
            if len(lk) != len(rk):
                raise ValueError("left_on and right_on must name the same "
                                 "number of columns")
        else:
            lk = rk = [c for c in left.columns if c in right.columns]
            inferred = True
            if not lk:
                raise ValueError(
                    "join() found no column in common - name the key with "
                    "on='id', or left_on=/right_on= when the two sides spell "
                    "it differently")
        for k in lk:
            if k not in left.columns:
                raise KeyError(f"join key {k!r} is not a column on the left "
                               f"(have: {', '.join(map(str, left.columns))})")
        for k in rk:
            if k not in right.columns:
                raise KeyError(f"join key {k!r} is not a column on the right "
                               f"(have: {', '.join(map(str, right.columns))})")

        # ---- types have to agree, and pandas' own error does not say how to fix it
        for a, b in zip(lk, rk):
            ka, kb = _dtype_kind(left[a]), _dtype_kind(right[b])
            if ka != kb:
                raise ValueError(
                    f"join key {a!r} is {ka} on the left but {b!r} is {kb} on "
                    f"the right - they can never match. Make them agree first, "
                    f"e.g. .to_float({a!r}) or .astype({a}='str')")

        # ---- who matched whom, measured before the merge so `how` can't hide
        # it. Only when someone is going to read it: on a big frame this walks
        # both key columns, and verbose=False should stay as cheap as pd.merge.
        if verbose:
            left_keys, right_keys = _key_tuples(left, lk), _key_tuples(right, rk)
            right_set, left_set = set(right_keys), set(left_keys)
            l_hit = [k in right_set for k in left_keys]
            r_hit = [k in left_set for k in right_keys]
            n_l_miss, n_r_miss = l_hit.count(False), r_hit.count(False)
            missed = sorted({k for k, hit in zip(left_keys, l_hit) if not hit},
                            key=lambda t: tuple(map(str, t)))

            # ---- would those misses match if the key were cleaned up?
            near = 0
            if missed and any(_dtype_kind(left[k]) == "text" for k in lk):
                folded_right = set(_folded_key_tuples(right, rk))
                miss_rows = left.loc[[not h for h in l_hit], lk]
                near = len({t for t in _folded_key_tuples(miss_rows, lk)
                            if t in folded_right})

        overlap = [c for c in right.columns
                   if c in left.columns and c not in (rk if lk == rk else [])]
        merged = (pd.merge(left, right, on=lk, how=how, suffixes=("", suffix))
                  if lk == rk else
                  pd.merge(left, right, left_on=lk, right_on=rk, how=how,
                           suffixes=("", suffix)))
        merged = merged.reset_index(drop=True)

        if verbose:
            key_txt = lk[0] if len(lk) == 1 else str(lk)
            print(f"\n{BOLD}{UNDER}JOIN{RESET}")
            print(f"  + {how} join on {key_txt!r}"
                  f"{' (inferred - the shared column)' if inferred else ''}")
            print(f"  + {l_hit.count(True)} of {len(left)} left rows matched, "
                  f"{r_hit.count(True)} of {len(right)} right rows used")
            if n_l_miss:
                fate = "dropped" if how in ("inner", "right") else "kept, with nulls"
                print(f"  ! {_rows(n_l_miss)} on the left matched nothing "
                      f"({fate}): {_fmt_keys(missed)}")
            if near:
                print(f"  ! {near} of those keys match after case/whitespace "
                      f"folding - .clean() both sides before joining")
            if n_r_miss:
                fate = "dropped" if how in ("inner", "left") else "kept, with nulls"
                print(f"  ! {_rows(n_r_miss)} on the right matched nothing ({fate})")
            n_null_keys = int(left[lk].isna().any(axis=1).sum())
            if n_null_keys:
                print(f"  ! {_rows(n_null_keys)} on the left have a null key")
            if len(merged) > len(left) and how in ("left", "inner"):
                print(f"  ! the right key is not unique - the join added "
                      f"{_rows(len(merged) - len(left))}")
            if overlap:
                print(f"  + right columns kept under a suffix: "
                      f"{', '.join(f'{c}{suffix}' for c in overlap)}")
            print(f"{BOLD}-> {left.shape[0]}x{left.shape[1]} to "
                  f"{merged.shape[0]}x{merged.shape[1]}{RESET}\n")

        key_repr = lk[0] if len(lk) == 1 else lk
        return self._derive(merged, step=f"join({how!r}, on={key_repr!r})")

    @_classorinstance
    def concat(self, *sources, source_col=None, verbose=True):
        """Stack tables on top of each other - files, globs, Data or frames.

            Data.concat("data/2024-*.csv")            # a folder of monthly files
            Data.concat("jan.csv", "feb.csv")
            d.concat(more)                            # onto an existing Data

        ``source_col="file"`` records which source each row came from, which is
        what you want the moment two files disagree.

        Columns are matched by name; the report calls out the two things that
        quietly ruin a stacked dataset - a column missing from one source (it
        becomes nulls) and a column whose type differs between sources (the
        result falls back to text). Returns a NEW ``Data``.
        """
        items = list(sources)
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            items = list(items[0])
        # a glob only earns its keep if it matches something
        expanded = []
        for item in items:
            if isinstance(item, str) and any(ch in item for ch in "*?["):
                hits = sorted(glob.glob(item))
                if not hits:
                    raise ValueError(f"no files match {item!r}")
                expanded.extend(hits)
            else:
                expanded.append(item)
        if self is not None:
            expanded.insert(0, self)
        if not expanded:
            raise ValueError("concat() needs at least one source")

        frames, labels = [], []
        for i, item in enumerate(expanded):
            frames.append(Data._as_frame(item))
            labels.append(Data._label(item, i))
        if self is not None:
            labels[0] = "this data"
        if source_col:
            frames = [f.assign(**{source_col: lab}) for f, lab in zip(frames, labels)]

        cols = []
        for f in frames:
            cols.extend(c for c in f.columns if c not in cols)
        out = pd.concat(frames, ignore_index=True, sort=False).reindex(columns=cols)

        if verbose:
            print(f"\n{BOLD}{UNDER}CONCAT{RESET}")
            print(f"  + {len(frames)} sources: "
                  f"{', '.join(f'{lab} ({_rows(len(f))})' for lab, f in zip(labels, frames))}")
            for c in cols:
                absent = [lab for lab, f in zip(labels, frames) if c not in f.columns]
                if absent:
                    print(f"  ! {c!r} is missing from {', '.join(absent)} "
                          f"- those rows are null there")
                kinds = {_dtype_kind(f[c]) for f in frames if c in f.columns}
                if len(kinds) > 1:
                    print(f"  ! {c!r} is {' / '.join(sorted(kinds))} in different "
                          f"sources - run .clean() on the result")
            n_dupes = int(out.duplicated().sum())
            if n_dupes:
                hint = "" if source_col else ", source_col='file' shows where from"
                print(f"  ! {_rows(n_dupes)} duplicated across sources "
                      f"- .dedupe() drops them{hint}")
            if source_col:
                print(f"  + tagged every row with its source in {source_col!r}")
            print(f"{BOLD}-> {out.shape[0]}x{out.shape[1]}{RESET}\n")

        step = f"concat({len(frames)} sources)"
        if self is not None:
            return self._derive(out, step=step)
        return Data(df=out, steps=[step])

    # ----------------------------------------------------------- AGGREGATE
    def groupby(self, *cols):
        return self._derive(self.df, group=list(cols))

    def agg(self, how, col=None):
        if not self._group:
            raise RuntimeError("Call groupby() before agg()")
        grp = self.df.groupby(self._group)
        if col is None:
            out = grp.agg(how)
        else:
            out = grp.agg({col: how}).reset_index()
        return self._derive(out, step=f"groupby({self._group}).agg({how!r}, {col!r})")

    def stat(self, how, col=None, by=None):
        """One-call aggregate - no separate ``groupby()`` step.

            d.stat("mean", "price", by="city")

        ``by`` may be a column or a list of them; omit it to aggregate the
        whole frame. The named shortcuts below read better in practice.
        """
        if by is not None:
            keys = [by] if isinstance(by, str) else list(by)
            grp = self.df.groupby(keys)
            if col is None:
                out = (grp.size().reset_index(name="count") if how == "count"
                       else grp.agg(how).reset_index())
            else:
                out = grp.agg({col: how}).reset_index()
        elif col is None:
            out = (pd.DataFrame([{"count": len(self.df)}]) if how == "count"
                   else self.df.select_dtypes(include="number").agg(how).to_frame().T)
        else:
            out = pd.DataFrame([{f"{how}_{col}": getattr(self.df[col], how)()}])
        return self._derive(out, step=f"stat({how!r}, {col!r}, by={by!r})",
                            call=("stat", (how,), {"col": col, "by": by}))

    def mean(self, col=None, by=None):
        """``d.mean("price", by="city")`` - mean of a column, optionally grouped."""
        return self.stat("mean", col, by)

    def sum(self, col=None, by=None):
        """``d.sum("revenue", by="city")``"""
        return self.stat("sum", col, by)

    def count(self, col=None, by=None):
        """``d.count(by="city")`` - rows per group (no dummy column needed)."""
        return self.stat("count", col, by)

    def median(self, col=None, by=None):
        """``d.median("price", by="city")``"""
        return self.stat("median", col, by)

    def min(self, col=None, by=None):
        """``d.min("price", by="city")``"""
        return self.stat("min", col, by)

    def max(self, col=None, by=None):
        """``d.max("price", by="city")``"""
        return self.stat("max", col, by)

    def top(self, n=5, by=None):
        """``d.top(5, "price")`` - the n highest rows by a column."""
        df = self.df if by is None else self.df.sort_values(by, ascending=False)
        return self._derive(df.head(n).reset_index(drop=True),
                            step=f"top({n}, {by!r})",
                            call=("top", (), {"n": n, "by": by}))

    def bottom(self, n=5, by=None):
        """``d.bottom(5, "price")`` - the n lowest rows by a column."""
        df = self.df if by is None else self.df.sort_values(by, ascending=True)
        return self._derive(df.head(n).reset_index(drop=True),
                            step=f"bottom({n}, {by!r})",
                            call=("bottom", (), {"n": n, "by": by}))

    def counts(self, col):
        """``d.counts("city")`` - frequency table for one column."""
        out = self.df[col].value_counts(dropna=False).reset_index()
        out.columns = [col, "count"]
        return self._derive(out, step=f"counts({col!r})",
                            call=("counts", (col,), {}))

    def summarize(self, **kwargs):
        """Quick named stats. summarize(mean_sal='mean(salary)', n='count()')"""
        out = {}
        for k, v in kwargs.items():
            m = re.match(r"(\w+)\(([\w ]*)\)", v)
            if m:
                func, col = m.group(1), m.group(2).strip()
                if col:
                    out[k] = getattr(self.df[col], func)()
                elif func == "count":
                    out[k] = len(self.df)
                else:
                    out[k] = getattr(self.df, func)()
        return self._derive(pd.DataFrame([out]), step=f"summarize({list(kwargs)})",
                            call=("summarize", (), kwargs))

    def corr(self, method="pearson"):
        """Return the correlation matrix as a DataFrame."""
        return self._derive(self.df.corr(numeric_only=True, method=method),
                            step=f"corr({method!r})",
                            call=("corr", (), {"method": method}))

    def plot_corr(self, title="Correlation matrix", cmap="coolwarm", show=None):
        """Heatmap of the numeric correlation matrix.

        Renders inline in a notebook; ``show=True`` forces a window, and
        ``show=False`` holds the figure for ``.savefig()``.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        c = self.df.corr(numeric_only=True)
        im = ax.imshow(c, cmap=cmap)
        ax.set_xticks(range(len(c.columns)))
        ax.set_yticks(range(len(c.columns)))
        ax.set_xticklabels(c.columns, rotation=45, ha="right")
        ax.set_yticklabels(c.columns)
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        self._fig = _render(fig, show)
        return self

    # ----------------------------------------------------------- VISUALIZE
    def plot(self, kind="line", x=None, y=None, title=None, show=None, **kwargs):
        """One-liner plot. kind: line|bar|hist|scatter|box|pie

        ``x`` and ``y`` are optional: on a two-column frame (what a grouped
        aggregate leaves you with) they are inferred, so a whole pipeline
        ends ``.mean("price", by="city").plot("bar")``.

        The chart shows up where you are working - inline in a notebook, no
        PNG round-trip. From a script or REPL, end the chain with ``.show()``
        to open it in a window (or pass ``show=True``), or ``.savefig(path)``
        to write it out. ``show=False`` always holds the figure back.
        """
        if x is None and y is None:
            x, y = self._infer_xy(kind)
        fig, ax = plt.subplots(figsize=(8, 5))
        if kind == "scatter":
            ax.scatter(self.df[x], self.df[y])
        elif kind == "hist":
            ax.hist(self.df[x or y], **kwargs)
        elif kind == "box":
            self.df.boxplot(column=y, by=x, ax=ax)
        elif kind == "pie":
            self.df.plot(kind="pie", y=y, labels=self.df[x], ax=ax, **kwargs)
        else:
            self.df.plot(kind=kind, x=x, y=y, ax=ax, **kwargs)
        if title:
            ax.set_title(title)
        self._fig = _render(fig, show)
        return self

    def _infer_xy(self, kind):
        """Guess the plot columns: label column vs first numeric column."""
        num = list(self.df.select_dtypes(include="number").columns)
        if kind == "hist":
            return (num[0] if num else None), None
        if not num:
            raise ValueError(
                "plot() found no numeric column to chart - name one with y=, "
                "or run .clean()/.to_float() first")
        other = [c for c in self.df.columns if c not in num]
        # y is the value column; x is the label column, or the index if there
        # is no separate label column (a one-column frame plots against index).
        return (other[0] if other else None), num[0]

    def show(self):
        """Display the current figure - or the data, if there is no figure.

        In a notebook the chart renders inline, in the same cell; from a script
        or REPL it opens in a window. On a machine with no display (CI, a
        server) it says so and points you at ``.savefig()``.
        """
        if self._fig is not None:
            _render(self._fig, True)
        else:
            print(self.df)
        return self

    def savefig(self, path):
        """Write the current figure to a file. For when you want the PNG."""
        if self._fig is not None:
            self._fig.savefig(path, bbox_inches="tight")
            print(f"saved plot -> {path}")
        else:
            raise RuntimeError("No figure to save. Call plot()/plot_corr() first.")
        return self

    # ----------------------------------------------------------- RECIPES
    def save_recipe(self, path):
        """Write this pipeline to a JSON recipe another file can replay.

        Output method: writes a file and returns the SAME object.

            Data("jan.csv").clean().filter("units > 5").save_recipe("m.json")

        Only whitelisted steps replay (see ``dclean.core.REPLAYABLE``). A step
        that cannot - `join()`, `concat()`, `groupby().agg()` - is refused HERE,
        naming the step, rather than dropped: a recipe that quietly skipped your
        join would produce a different dataset without saying so.
        """
        steps = []
        for i, s in enumerate(self._steps, 1):
            method = s["method"]
            if method not in REPLAYABLE:
                hint = _NOT_REPLAYABLE_HINT.get(
                    str(s["display"]).split("(")[0].split(".")[0], "")
                raise ValueError(
                    "step {} ({}) cannot be replayed{}. Remove it from the "
                    "pipeline you save the recipe from, or save the recipe "
                    "before it.".format(i, s["display"], " - " + hint if hint else ""))
            for name, value in list(s["kwargs"].items()) + list(enumerate(s["args"])):
                if not _jsonable(value):
                    raise ValueError(
                        "step {} ({}) has an argument that is not JSON - {!r}. "
                        "A recipe is a plain file; pass column names and "
                        "literals, not objects.".format(i, s["display"], value))
            steps.append({"method": method, "args": s["args"],
                          "kwargs": s["kwargs"], "display": s["display"]})

        recipe = {
            "recipe": RECIPE_FORMAT,
            "dcleaner": _version(),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "steps": steps,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(recipe, fh, indent=2)
            fh.write("\n")
        print(f"saved recipe -> {path} ({len(steps)} steps)")
        return self

    def apply_recipe(self, path, verbose=True):
        """Replay a saved recipe onto this dataset. Returns a NEW ``Data``.

        Transform: every step in the recipe runs in order, exactly as if you
        had typed the calls, so the result carries the whole pipeline in its
        own ``log()``::

            Data("feb.csv").apply_recipe("monthly.json")

        Only whitelisted methods run - the check is repeated here because the
        file may have been edited since it was written - and an unknown method
        is an error, never a silent skip.

        SECURITY: `filter()` and `mutate()` steps are expressions that get
        evaluated, so a recipe file is as trusted as a Python script. Run
        recipes you wrote, not recipes you were sent.
        """
        with open(path, encoding="utf-8") as fh:
            recipe = json.load(fh)
        if not isinstance(recipe, dict) or "steps" not in recipe:
            raise ValueError(f"{path} is not a dclean recipe (no 'steps' key)")
        fmt = recipe.get("recipe")
        if fmt != RECIPE_FORMAT:
            raise ValueError(
                f"{path} is recipe format {fmt!r}, this dcleaner reads "
                f"{RECIPE_FORMAT}. Re-save it with the version that wrote it.")

        out = self
        for i, s in enumerate(recipe["steps"], 1):
            method = s.get("method")
            if method not in REPLAYABLE:
                raise ValueError(
                    "step {} of {} calls {!r}, which dclean will not replay. "
                    "Replayable methods: {}".format(
                        i, path, method, ", ".join(sorted(REPLAYABLE))))
            args = list(s.get("args") or [])
            kwargs = dict(s.get("kwargs") or {})
            try:
                out = getattr(out, method)(*args, **kwargs)
            except Exception as e:
                raise ValueError(
                    "step {} of {} - {}({}) - failed on this data: {}".format(
                        i, path, method,
                        ", ".join([repr(a) for a in args]
                                  + [f"{k}={v!r}" for k, v in kwargs.items()]),
                        e))
        if verbose:
            print(f"{BOLD}-> replayed {len(recipe['steps'])} steps from "
                  f"{path}{RESET}")
        return out

    # ----------------------------------------------------------- EXPORT
    def to_csv(self, path):
        """Write the frame to CSV, no index column.

        Output method: writes a file and returns the SAME object, so an export
        can sit mid-chain.
        """
        self.df.to_csv(path, index=False)
        print(f"saved -> {path}")
        return self

    def to_excel(self, path, sheet_name="Sheet1"):
        """Write the frame to an Excel workbook, no index column.

        Output method: writes a file and returns the SAME object.

        Needs ``openpyxl``, which dclean does not install for you
        (``pip install openpyxl``); the error says so if it is missing.
        """
        _require("openpyxl", "to_excel()")
        self.df.to_excel(path, sheet_name=sheet_name, index=False)
        print(f"saved -> {path}")
        return self

    def to_json(self, path, orient="records", indent=2):
        """Write the frame to JSON.

        Output method: writes a file and returns the SAME object.

        Defaults to ``orient="records"`` - a list of row objects, the shape
        most tools expect - and ISO-8601 dates, so a column ``clean()`` parsed
        into datetimes reads back as a date rather than epoch milliseconds.
        """
        self.df.to_json(path, orient=orient, indent=indent, date_format="iso")
        print(f"saved -> {path}")
        return self

    def to_parquet(self, path, **kwargs):
        """Write the frame to Parquet.

        Output method: writes a file and returns the SAME object.

        Needs ``pyarrow`` or ``fastparquet``, neither of which dclean installs
        for you (``pip install pyarrow``); the error says so if both are
        missing. Extra keyword arguments go straight to pandas
        (``compression=``, ``engine=`` ...).
        """
        _require(("pyarrow", "fastparquet"), "to_parquet()", install="pyarrow")
        self.df.to_parquet(path, index=False, **kwargs)
        print(f"saved -> {path}")
        return self

    def to_df(self):
        """Hand back the raw DataFrame for full pandas power."""
        return self.df

    def to_fig(self):
        """Hand back the matplotlib Figure for full matplotlib power."""
        if self._fig is None:
            raise RuntimeError("No figure yet. Call plot()/plot_corr() first.")
        return self._fig

    def copy(self):
        """An independent copy (rarely needed - transforms already copy)."""
        return self._derive(self.df.copy(), step="copy()", call=("copy", (), {}))

    # ----------------------------------------------------------- DUNDERS
    def __str__(self):
        """`print(d)` shows the full dataset (not just the terse repr)."""
        return tabulate(self.df, headers="keys", tablefmt="github", showindex=False)

    def __repr__(self):
        cols = ", ".join(map(str, self.df.columns)) if len(self.df.columns) else "-"
        return f"dclean.Data({self.df.shape[0]}x{self.df.shape[1]}, cols=[{cols}])"

    def _repr_html_(self, n=10):
        """Rich table for Jupyter - a display hook, so it changes nothing.

        A notebook renders this in place of ``__repr__``, so a bare ``d`` at
        the end of a cell shows the data instead of only its shape. ``repr()``
        and ``str()`` are deliberately left alone: the terminal keeps the terse
        one-liner and ``print(d)`` keeps the full table.
        """
        rows, cols = self.df.shape
        header = f"{rows} rows &times; {cols} cols"
        if rows > n:
            header += f" &mdash; showing the first {n}"
        return (f'<div style="font-family:monospace;font-size:0.9em;'
                f'margin-bottom:0.4em"><b>dclean.Data</b> &mdash; {header}</div>'
                f'{self.df.head(n).to_html()}')

    def __len__(self):
        return len(self.df)
