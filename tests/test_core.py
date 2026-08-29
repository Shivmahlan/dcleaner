import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest

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


def test_nulls_reports_total(capsys):
    d = Data(SAMPLE).dropna()
    d.nulls()
    out = capsys.readouterr().out
    assert "total nulls" in out


def test_to_float_coerces():
    # Build a frame with a string-number column and a clean one.
    # Transforms are immutable, so bind the result.
    d = Data.from_records([
        {"x": "1.5", "y": "2"},
        {"x": "bad", "y": "3"},
    ]).to_float("x", "y")
    assert pd.api.types.is_float_dtype(d.df["x"])
    assert pd.isna(d.df["x"].iloc[1])   # "bad" -> NaN
    assert d.df["y"].iloc[1] == 3.0


def test_to_table_prints_all_columns(capsys):
    d = Data(SAMPLE)
    d.to_table()
    out = capsys.readouterr().out
    # every column header should appear in the printed table
    for col in d.df.columns:
        assert col in out


def test_repr_shows_shape_and_cols():
    d = Data(SAMPLE)
    r = repr(d)
    assert "dclean.Data(" in r
    assert "cols=" in r
    assert str(len(d.df.columns)) in r


def test_dtypes_prints_column_types(capsys):
    d = Data(SAMPLE)
    d.dtypes()
    out = capsys.readouterr().out
    # every column name must appear in the dtype dump
    for col in d.df.columns:
        assert col in out
    # and at least one pandas dtype string
    assert any(t in out for t in ("int64", "float64", "object"))


def test_shape_shows_columns_and_dtypes(capsys):
    d = Data(SAMPLE)
    d.shape()
    out = capsys.readouterr().out
    assert "rows x" in out
    assert "columns:" in out
    assert "dtypes:" in out
    for col in d.df.columns:
        assert col in out


def test_print_full_dataset(capsys):
    d = Data(SAMPLE)
    d.print()
    out = capsys.readouterr().out
    # full dataset prints every column header
    for col in d.df.columns:
        assert col in out


def test_print_caps_rows(capsys):
    d = Data(SAMPLE)
    d.print(n=3)
    out = capsys.readouterr().out
    # tablefmt=github body rows = data rows; capped to 3
    assert out.count("|") >= 4  # header + 3 data rows each have pipes
    # ensure all columns still shown in header
    for col in d.df.columns:
        assert col in out


def test_str_prints_dataset(capsys):
    # print(d) should render the dataset, not the terse repr
    d = Data(SAMPLE)
    print(d)
    out = capsys.readouterr().out
    for col in d.df.columns:
        assert col in out
    assert "dclean.Data(" not in out  # str != repr


# --------------------------------------------------------------- IMMUTABILITY

def test_transforms_do_not_mutate_source():
    raw = Data.from_records([{"a": 1}, {"a": None}, {"a": 3}])
    cleaned = raw.dropna()
    assert len(raw) == 3          # source untouched
    assert len(cleaned) == 2
    assert cleaned is not raw


def test_every_transform_returns_new_object():
    d = Data.from_records([{"a": 1, "b": "2"}, {"a": 2, "b": "3"}])
    for call in (lambda x: x.dropna(),
                 lambda x: x.fillna(0),
                 lambda x: x.drop("b"),
                 lambda x: x.keep("a"),
                 lambda x: x.rename(a="z"),
                 lambda x: x.dedupe(),
                 lambda x: x.lower_cols(),
                 lambda x: x.to_float("b"),
                 lambda x: x.filter("a > 0"),
                 lambda x: x.mutate(c="a + 1"),
                 lambda x: x.select("a"),
                 lambda x: x.sort("a"),
                 lambda x: x.groupby("b").agg("mean", "a"),
                 lambda x: x.summarize(n="count()"),
                 lambda x: x.corr(),
                 lambda x: x.fix_nulls(),
                 lambda x: x.clean(verbose=False)):
        assert call(d) is not d


def test_mutate_leaves_source_columns_alone():
    raw = Data.from_records([{"a": 1}])
    raw.mutate(b="a * 2")
    assert "b" not in raw.to_df().columns


def test_intermediate_variables_keep_their_value():
    step1 = Data.from_records([{"a": 1}, {"a": 2}, {"a": 3}])
    step2 = step1.filter("a > 1")
    step3 = step2.filter("a > 2")
    assert (len(step1), len(step2), len(step3)) == (3, 2, 1)


# --------------------------------------------------------------- BUNDLED DATA

def test_bundled_sample_loads_by_bare_name():
    d = Data("sample_sales.csv")
    assert len(d) > 0
    assert "Units" in d.to_df().columns


def test_samples_lists_bundled_files():
    assert "sample_sales.csv" in Data.samples()


# --------------------------------------------------------------- TOOLKIT

def test_clean_handles_everything_in_one_call():
    d = Data("sample_sales.csv").clean(verbose=False)
    df = d.to_df()
    # column names normalized
    assert "order_id" in df.columns and "unit_price" in df.columns
    # all-empty column dropped
    assert "legacy_column" not in df.columns
    # money text parsed to numbers
    assert pd.api.types.is_numeric_dtype(df["unit_price"])
    # dates parsed
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    # duplicates gone
    assert not df.duplicated().any()
    # whitespace trimmed
    assert not df["region"].dropna().str.startswith(" ").any()


def test_clean_nulls_strategies():
    assert Data("sample_sales.csv").clean(nulls="drop", verbose=False)\
        .to_df().isna().sum().sum() == 0
    assert Data("sample_sales.csv").clean(nulls="fill", verbose=False)\
        .to_df()["units"].isna().sum() == 0


def test_clean_rejects_bad_null_strategy():
    with pytest.raises(ValueError):
        Data("sample_sales.csv").clean(nulls="nonsense", verbose=False)


def test_clean_strips_currency_and_separators():
    d = Data.from_records([{"amt": "$1,234.50"}, {"amt": "$2,000.00"}])\
        .clean(verbose=False)
    assert d.to_df()["amt"].tolist() == [1234.5, 2000.0]


def test_clean_turns_na_tokens_into_real_nulls():
    d = Data.from_records([{"id": 1, "k": "a"},
                           {"id": 2, "k": "N/A"},
                           {"id": 3, "k": "null"}]).clean(verbose=False)
    assert d.to_df()["k"].isna().sum() == 2


def test_report_prints_profile_and_warnings(capsys):
    Data("sample_sales.csv").report()
    out = capsys.readouterr().out
    assert "DATASET REPORT" in out
    assert "duplicate rows" in out
    assert "warnings" in out


def test_fix_nulls_fills_numeric_and_text():
    d = Data.from_records([{"n": 1.0, "s": "x"}, {"n": None, "s": None},
                           {"n": 3.0, "s": "x"}]).fix_nulls()
    assert d.to_df().isna().sum().sum() == 0


def test_drop_outliers_removes_extremes():
    rows = [{"v": float(i)} for i in range(20)] + [{"v": 10000.0}]
    d = Data.from_records(rows).drop_outliers()
    assert 10000.0 not in d.to_df()["v"].tolist()
    assert len(d) == 20


def test_log_records_pipeline_steps():
    d = (Data("sample_sales.csv").clean(verbose=False)
         .filter("units > 5").sort("units"))
    steps = d.steps()
    assert len(steps) == 3
    assert any("clean" in s for s in steps)
    assert any("filter" in s for s in steps)


def test_describe_survives_non_numeric_frame(capsys):
    Data.from_records([{"city": "NY"}, {"city": "SF"}]).describe()
    out = capsys.readouterr().out
    assert "SUMMARY" in out          # no KeyError on 'mean'


def test_module_level_one_call_helpers(tmp_path):
    import dclean
    out = tmp_path / "clean.csv"
    d = dclean.clean("sample_sales.csv", to=str(out), verbose=False)
    assert out.exists()
    assert "order_id" in d.to_df().columns
    assert isinstance(dclean.load("sample_sales.csv"), Data)


# --------------------------------------------------------------- MINIMAL CALLS

def test_one_call_group_stats_match_groupby_agg():
    d = Data("sample_sales.csv").clean(verbose=False)
    short = d.mean("unit_price", by="city").to_df()
    long = d.groupby("city").agg("mean", "unit_price").to_df()
    pd.testing.assert_frame_equal(short, long)


def test_count_by_needs_no_dummy_column():
    d = Data.from_records([{"c": "a"}, {"c": "a"}, {"c": "b"}]).count(by="c")
    out = d.to_df().set_index("c")["count"].to_dict()
    assert out == {"a": 2, "b": 1}


def test_stat_shortcuts_on_whole_frame():
    d = Data.from_records([{"v": 2.0}, {"v": 4.0}])
    assert d.mean("v").to_df()["mean_v"].iloc[0] == 3.0
    assert d.sum("v").to_df()["sum_v"].iloc[0] == 6.0
    assert d.max("v").to_df()["max_v"].iloc[0] == 4.0
    assert d.count().to_df()["count"].iloc[0] == 2


def test_top_and_bottom():
    d = Data.from_records([{"v": i} for i in range(10)])
    assert d.top(3, "v").to_df()["v"].tolist() == [9, 8, 7]
    assert d.bottom(2, "v").to_df()["v"].tolist() == [0, 1]


def test_counts_frequency_table():
    d = Data.from_records([{"c": "x"}, {"c": "x"}, {"c": "y"}]).counts("c")
    assert d.to_df().columns.tolist() == ["c", "count"]
    assert d.to_df()["count"].tolist() == [2, 1]


def test_plot_infers_x_and_y(tmp_path):
    out = tmp_path / "p.png"
    (Data("sample_sales.csv").clean(verbose=False)
        .mean("unit_price", by="city")
        .plot("bar")               # no x=/y= needed
        .savefig(str(out)))
    assert out.exists() and os.path.getsize(out) > 0


def test_mutate_supports_accessor_expressions():
    d = Data.from_records([{"s": " A "}, {"s": "b"}])
    assert d.mutate(t="s.str.strip().str.lower()").to_df()["t"].tolist() == ["a", "b"]
    # the fast eval path still works for arithmetic
    d2 = Data.from_records([{"a": 2, "b": 3}])
    assert d2.mutate(c="a * b").to_df()["c"].iloc[0] == 6


# --------------------------------------------------------------- EDGE CASES

def test_zscore_outliers_keep_constant_column():
    # a column with no spread has no outliers - it must not wipe the frame
    d = Data.from_records([{"v": 5.0} for _ in range(6)])
    assert len(d.drop_outliers(method="zscore")) == 6
    assert len(d.drop_outliers(method="iqr")) == 6


def test_whole_frame_stats_ignore_text_columns():
    d = Data.from_records([{"c": "a", "v": 1.0}, {"c": "b", "v": 3.0}])
    assert d.mean().to_df()["v"].iloc[0] == 2.0


def test_plot_infers_y_on_single_column_frame(tmp_path):
    out = tmp_path / "p.png"
    Data.from_records([{"v": 1}, {"v": 2}]).plot("bar").savefig(str(out))
    assert out.exists()


def test_plot_without_numeric_column_explains_itself():
    with pytest.raises(ValueError, match="no numeric column"):
        Data.from_records([{"c": "a"}]).plot("bar")


def test_fix_nulls_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy must be one of"):
        Data.from_records([{"v": 1.0}, {"v": None}]).fix_nulls("bogus")


def test_empty_frame_survives_every_inspector(capsys):
    d = Data()
    d.clean(verbose=False).report()
    d.describe()
    d.nulls()
    assert "dataset is empty" in capsys.readouterr().out


# --------------------------------------------------------------- DATA SAFETY

def test_clean_reports_values_it_destroys(capsys):
    # 92% parse as numbers, so the column converts - the other 8 are real
    # values that become NaN. That loss must be stated, never silent.
    rows = ([{"amount": f"${i}.00"} for i in range(92)]
            + [{"amount": t} for t in ["pending", "refunded", "void", "TBD",
                                       "see note", "misc", "waived", "n/k"]])
    Data.from_records(rows).clean()
    out = capsys.readouterr().out
    assert "8 value(s) in 'amount' could not be parsed" in out


def test_clean_does_not_mutate_the_callers_dataframe():
    df = pd.DataFrame({"A ": [1, None, 3]})
    snapshot = df.copy()
    Data(df).clean(verbose=False).fix_nulls()
    pd.testing.assert_frame_equal(df, snapshot)


def test_report_can_hide_real_values():
    d = Data.from_records([{"email": "alice@example.com"}])
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        d.report(examples=False)
    out = buf.getvalue()
    assert "alice@example.com" not in out
    assert "<str>" in out
    # the default still shows a real example, which is the useful behaviour
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        d.report()
    assert "alice@example.com" in buf.getvalue()


def test_expression_guard_blocks_dunder_access():
    d = Data.from_records([{"n": 1}, {"n": 2}])
    for bad in ("n.__class__", "n.__class__.__base__"):
        with pytest.raises(ValueError, match="__"):
            d.filter(bad)
        with pytest.raises(ValueError, match="__"):
            d.mutate(z=bad)
    # ordinary expressions are untouched
    assert len(d.filter("n > 1")) == 1
    assert d.mutate(z="n * 2").to_df()["z"].tolist() == [2, 4]
