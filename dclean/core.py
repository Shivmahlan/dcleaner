"""Core fluent DataFrame wrapper for dclean."""
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe; plots still save/show in notebooks
import matplotlib.pyplot as plt


class Data:
    """Fluent wrapper around a pandas DataFrame.

    Every transform method returns ``self`` so calls chain:

        from dclean import Data
        (Data("sales.csv")
            .dropna()
            .filter("age > 18")
            .groupby("city").agg("mean", "salary")
            .plot("bar", x="city", y="salary")
            .show())

    Drop back to raw pandas anytime with ``.to_df()``.
    """

    def __init__(self, source=None, df=None):
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

    # ---------------------------------------------------------------- LOAD
    @staticmethod
    def _load(path):
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

    # -------------------------------------------------------------- INSPECT
    def head(self, n=5):
        print(self.df.head(n))
        return self

    def tail(self, n=5):
        print(self.df.tail(n))
        return self

    def info(self):
        self.df.info()
        return self

    def shape(self):
        print(f"{self.df.shape[0]} rows x {self.df.shape[1]} cols")
        return self

    def cols(self):
        print(list(self.df.columns))
        return self

    def describe(self):
        """Pretty summary of numeric columns."""
        print(self.df.describe())
        return self

    # --------------------------------------------------------------- CLEAN
    def dropna(self, subset=None):
        self.df = self.df.dropna(subset=subset)
        return self

    def fillna(self, value):
        self.df = self.df.fillna(value)
        return self

    def drop(self, cols):
        cols = [cols] if isinstance(cols, str) else list(cols)
        self.df = self.df.drop(columns=cols)
        return self

    def keep(self, cols):
        cols = [cols] if isinstance(cols, str) else list(cols)
        self.df = self.df[cols]
        return self

    def rename(self, **kwargs):
        self.df = self.df.rename(columns=kwargs)
        return self

    def dedupe(self, subset=None):
        self.df = self.df.drop_duplicates(subset=subset)
        return self

    def astype(self, **kwargs):
        self.df = self.df.astype(kwargs)
        return self

    def lower_cols(self):
        """Rename all columns to lowercase (common cleaning step)."""
        self.df = self.df.rename(columns={c: str(c).lower() for c in self.df.columns})
        return self

    # -------------------------------------------------------------- FILTER
    def filter(self, expr):
        """Filter with a readable expression string.

        Supports: == != > < >= <= and or in not in
        e.g.  filter("age > 18 and city in ['NY','LA']")
        Columns with spaces must be quoted: filter("'total sales' > 100")
        """
        self.df = self.df[self._eval_expr(expr)]
        return self

    def _eval_expr(self, expr):
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
        for col, expr in kwargs.items():
            self.df[col] = self.df.eval(expr) if isinstance(expr, str) else expr
        return self

    def select(self, *cols):
        self.df = self.df[list(cols)]
        return self

    def sort(self, by, ascending=True):
        self.df = self.df.sort_values(by, ascending=ascending)
        return self

    # ----------------------------------------------------------- AGGREGATE
    def groupby(self, *cols):
        self._group = list(cols)
        return self

    def agg(self, how, col=None):
        if not self._group:
            raise RuntimeError("Call groupby() before agg()")
        grp = self.df.groupby(self._group)
        if col is None:
            self.df = grp.agg(how)
        else:
            self.df = grp.agg({col: how}).reset_index()
        self._group = None
        return self

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
        self.df = pd.DataFrame([out])
        return self

    def corr(self, method="pearson"):
        """Return the correlation matrix as a DataFrame."""
        self.df = self.df.corr(numeric_only=True, method=method)
        return self

    def plot_corr(self, title="Correlation matrix", cmap="coolwarm"):
        """Heatmap of the numeric correlation matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        c = self.df.corr(numeric_only=True)
        im = ax.imshow(c, cmap=cmap)
        ax.set_xticks(range(len(c.columns)))
        ax.set_yticks(range(len(c.columns)))
        ax.set_xticklabels(c.columns, rotation=45, ha="right")
        ax.set_yticklabels(c.columns)
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        self._fig = fig
        return self

    # ------------------------------------------------------------ VISUALIZE
    def plot(self, kind="line", x=None, y=None, title=None, **kwargs):
        """One-liner plot. kind: line|bar|hist|scatter|box|pie"""
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
        self._fig = fig
        return self

    def show(self):
        if self._fig is not None:
            plt.show()
        else:
            print(self.df)
        return self

    def savefig(self, path):
        if self._fig is not None:
            self._fig.savefig(path, bbox_inches="tight")
            print(f"saved plot -> {path}")
        else:
            raise RuntimeError("No figure to save. Call plot()/plot_corr() first.")
        return self

    # --------------------------------------------------------------- EXPORT
    def to_csv(self, path):
        self.df.to_csv(path, index=False)
        print(f"saved -> {path}")
        return self

    def to_df(self):
        """Hand back the raw DataFrame for full pandas power."""
        return self.df

    # ----------------------------------------------------------- DUNDERS
    def __repr__(self):
        return f"Data(shape={self.df.shape})"

    def __len__(self):
        return len(self.df)
