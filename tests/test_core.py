import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from dclean import Data

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "data", "sample.csv")


def test_load_and_shape():
    d = Data(SAMPLE)
    assert len(d) == 60
    assert d.df.shape[1] == 4


def test_dropna_removes_nulls():
    d = Data(SAMPLE).dropna()
    assert d.df.isna().sum().sum() == 0
    # we injected 4 NaN salaries + 1 NaN age -> at least 4 rows dropped
    assert len(d) <= 56


def test_filter_expression():
    d = Data(SAMPLE).dropna().filter("age > 18")
    assert (d.df["age"] > 18).all()


def test_groupby_agg():
    d = (Data(SAMPLE).dropna()
         .groupby("city").agg("mean", "salary"))
    assert "salary" in d.df.columns
    assert "city" in d.df.columns
    assert len(d.df) == 3  # NY, LA, SF


def test_mutate_new_column():
    d = Data(SAMPLE).dropna().mutate(double_salary="salary * 2")
    assert "double_salary" in d.df.columns
    assert (d.df["double_salary"] == d.df["salary"] * 2).all()


def test_corr_returns_matrix():
    d = Data(SAMPLE).dropna().corr()
    assert isinstance(d.df, pd.DataFrame)
    assert d.df.shape[0] == d.df.shape[1]


def test_plot_savefig(tmp_path):
    out = tmp_path / "plot.png"
    (Data(SAMPLE).dropna()
        .groupby("city").agg("mean", "salary")
        .plot("bar", x="city", y="salary")
        .savefig(str(out)))
    assert out.exists()
    assert os.path.getsize(out) > 0


def test_to_df_returns_pandas():
    d = Data(SAMPLE)
    raw = d.to_df()
    assert isinstance(raw, pd.DataFrame)
    assert raw.shape == d.df.shape
