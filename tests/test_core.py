import os
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest

from dclean import Data, core

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


# ----------------------------------------------------------------- DISPLAY

@pytest.mark.parametrize("backend,mode", [
    ("module://matplotlib_inline.backend_inline", "notebook"),
    ("nbAgg", "notebook"),
    ("module://ipympl.backend_nbagg", "notebook"),
    ("Agg", "headless"),
    ("svg", "headless"),
    ("TkAgg", "gui"),
    ("MacOSX", "gui"),
    ("QtAgg", "gui"),
])
def test_display_mode_classifies_backends(monkeypatch, backend, mode):
    monkeypatch.setattr(core.matplotlib, "get_backend", lambda: backend)
    assert core._display_mode() == mode


def test_import_never_pins_the_backend():
    # The whole point: dclean must not force Agg, or plots can only ever be
    # looked at as a saved PNG. An explicit choice must survive the import.
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    code = ("import matplotlib; matplotlib.use('svg');"
            "import dclean; print(matplotlib.get_backend())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, cwd=os.path.dirname(HERE))
    assert out.stdout.strip().lower() == "svg", out.stderr


def test_plot_renders_inline_and_leaves_no_duplicate_figure():
    pytest.importorskip("matplotlib_inline")
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    code = ("import matplotlib;"
            "matplotlib.use('module://matplotlib_inline.backend_inline');"
            "import matplotlib.pyplot as plt; from dclean import Data;"
            "d = Data.from_records([{'v': 1}, {'v': 2}]).plot('bar');"
            "print('OPEN', plt.get_fignums());"
            "print('FIG', type(d.to_fig()).__name__)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, cwd=os.path.dirname(HERE))
    # displayed once, then closed - otherwise the inline backend flushes it
    # again at the end of the cell and the notebook shows the chart twice
    assert "OPEN []" in out.stdout, out.stderr
    assert "FIG Figure" in out.stdout, out.stderr


def test_show_without_a_display_explains_itself(capsys):
    # tests run under Agg; show() used to be a silent no-op here
    Data.from_records([{"v": 1}, {"v": 2}]).plot("bar").show()
    out = capsys.readouterr().out
    assert "no display available" in out and "savefig" in out


def test_show_without_a_figure_prints_the_data(capsys):
    Data.from_records([{"v": 1}, {"v": 2}]).show()
    assert "v" in capsys.readouterr().out


def test_plot_show_false_still_saves(tmp_path):
    out = tmp_path / "p.png"
    Data.from_records([{"v": 1}, {"v": 2}]).plot("bar", show=False).savefig(str(out))
    assert out.exists() and os.path.getsize(out) > 0


def test_to_fig_requires_a_plot():
    with pytest.raises(RuntimeError, match="No figure"):
        Data.from_records([{"v": 1}]).to_fig()


# ----------------------------------------------------------------- COMBINE

def _regions():
    return Data.from_records([
        {"city": "NY", "region": "east"},
        {"city": "LA", "region": "west"},
        {"city": "SF", "region": "west"},
    ])


def _sales():
    return Data.from_records([
        {"city": "NY", "units": 1},
        {"city": "LA", "units": 2},
        {"city": "SF ", "units": 3},     # padded - looks matched, is not
        {"city": "boston", "units": 4},  # genuinely absent on the right
    ])


def test_join_keeps_every_left_row_and_counts_the_matches(capsys):
    out = _sales().join(_regions(), on="city")
    assert len(out) == 4                          # left join keeps them all
    assert out.to_df()["region"].isna().sum() == 2
    printed = capsys.readouterr().out
    assert "2 of 4 left rows matched" in printed
    assert "2 rows on the left matched nothing" in printed
    assert "'SF '" in printed and "'boston'" in printed


def test_join_flags_keys_that_only_need_cleaning(capsys):
    # the quiet killer: 'SF ' vs 'SF' matches nothing and nobody notices
    _sales().join(_regions(), on="city")
    printed = capsys.readouterr().out
    assert "1 of those keys match after case/whitespace folding" in printed


def test_join_inner_drops_the_misses(capsys):
    out = _sales().join(_regions(), on="city", how="inner")
    assert len(out) == 2
    assert "(dropped)" in capsys.readouterr().out


def test_join_warns_when_the_right_key_multiplies_rows(capsys):
    reps = Data.from_records([{"city": "NY", "rep": "ann"},
                              {"city": "NY", "rep": "bob"}])
    out = _sales().join(reps, on="city")
    assert len(out) == 5                          # 4 left rows, one duplicated
    assert "the right key is not unique - the join added 1 row" in capsys.readouterr().out


def test_join_warns_about_null_keys(capsys):
    left = Data.from_records([{"city": "NY", "units": 1}, {"city": None, "units": 2}])
    left.join(_regions(), on="city")
    assert "1 row on the left have a null key" in capsys.readouterr().out


def test_join_infers_the_shared_column(capsys):
    out = _sales().join(_regions())
    assert "region" in out.to_df().columns
    assert "(inferred - the shared column)" in capsys.readouterr().out


def test_join_without_a_shared_column_says_what_to_do():
    with pytest.raises(ValueError, match="no column in common"):
        _sales().join(Data.from_records([{"zone": "a"}]))


def test_join_rejects_keys_that_can_never_match():
    numeric_key = Data.from_records([{"city": 1, "region": "east"}])
    with pytest.raises(ValueError, match="text on the left but .* is number"):
        _sales().join(numeric_key, on="city")


def test_join_accepts_differently_named_keys():
    right = Data.from_records([{"name": "NY", "region": "east"}])
    out = _sales().join(right, left_on="city", right_on="name", verbose=False)
    assert out.to_df()["region"].tolist()[0] == "east"
    with pytest.raises(ValueError, match="must be given together"):
        _sales().join(right, left_on="city")


def test_join_suffixes_overlapping_columns(capsys):
    right = Data.from_records([{"city": "NY", "units": 99}])
    out = _sales().join(right, on="city")
    assert out.to_df()["units"].tolist()[0] == 1          # left column wins its name
    assert out.to_df()["units_right"].tolist()[0] == 99
    assert "units_right" in capsys.readouterr().out


def test_join_reads_a_path_or_a_dataframe(tmp_path):
    csv = tmp_path / "regions.csv"
    _regions().to_csv(str(csv))
    from_path = _sales().join(str(csv), on="city", verbose=False)
    from_frame = _sales().join(_regions().to_df(), on="city", verbose=False)
    pd.testing.assert_frame_equal(from_path.to_df(), from_frame.to_df())
    with pytest.raises(TypeError, match="expected a Data"):
        _sales().join(42)


def test_join_rejects_an_unknown_how():
    with pytest.raises(ValueError, match="how must be"):
        _sales().join(_regions(), on="city", how="sideways")


def test_join_never_touches_either_side(capsys):
    left, right = _sales(), _regions()
    before_l, before_r = left.to_df().copy(), right.to_df().copy()
    left.join(right, on="city")
    pd.testing.assert_frame_equal(left.to_df(), before_l)
    pd.testing.assert_frame_equal(right.to_df(), before_r)


def test_join_is_silent_and_logged(capsys):
    out = _sales().join(_regions(), on="city", verbose=False)
    assert capsys.readouterr().out == ""
    assert out.steps() == ["join('left', on='city')"]


def test_concat_stacks_files_from_a_glob(tmp_path, capsys):
    for name, city in [("m_01.csv", "NY"), ("m_02.csv", "LA")]:
        Data.from_records([{"city": city, "units": 1}]).to_csv(str(tmp_path / name))
    out = Data.concat(str(tmp_path / "m_*.csv"), source_col="file")
    assert len(out) == 2
    assert out.to_df()["file"].tolist() == ["m_01.csv", "m_02.csv"]
    assert "2 sources" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no files match"):
        Data.concat(str(tmp_path / "nothing_*.csv"))


def test_concat_on_an_instance_keeps_its_own_rows(capsys):
    a = Data.from_records([{"v": 1}])
    out = a.concat(Data.from_records([{"v": 2}]))
    assert out.to_df()["v"].tolist() == [1, 2]     # not just [2]
    assert "this data" in capsys.readouterr().out
    assert len(a) == 1                             # and `a` itself is untouched


def test_concat_takes_a_list_and_records_the_step():
    out = Data.concat([Data.from_records([{"v": 1}]), Data.from_records([{"v": 2}])],
                      verbose=False)
    assert len(out) == 2
    assert out.steps() == ["concat(2 sources)"]


def test_concat_reports_columns_that_do_not_line_up(capsys):
    Data.concat(Data.from_records([{"a": 1, "notes": "x"}]),
                Data.from_records([{"a": 2, "extra": 9}]))
    printed = capsys.readouterr().out
    assert "'notes' is missing from source 1" in printed
    assert "'extra' is missing from source 0" in printed


def test_concat_reports_a_type_that_changes_between_sources(capsys):
    Data.concat(Data.from_records([{"price": 1.5}]),
                Data.from_records([{"price": "$3.50"}]))
    assert "'price' is number / text in different sources" in capsys.readouterr().out


def test_concat_flags_rows_duplicated_across_sources(capsys):
    row = [{"v": 1}]
    out = Data.concat(Data.from_records(row), Data.from_records(row))
    assert "1 row duplicated across sources" in capsys.readouterr().out
    assert len(out.dedupe()) == 1


def test_concat_refuses_to_overwrite_an_existing_source_col():
    with pytest.raises(ValueError, match="already a column"):
        Data.concat(Data.from_records([{"v": 1, "file": "keep me"}]),
                    Data.from_records([{"v": 2}]), source_col="file")


def test_the_bundled_lookup_joins_to_the_bundled_sales(capsys):
    # the README's join example must keep working after a plain `pip install`
    assert "sample_cities.csv" in Data.samples()
    out = (Data("sample_sales.csv").clean(verbose=False)
           .join("sample_cities.csv", on="city"))
    assert "population" in out.to_df().columns
    assert "region_right" in out.to_df().columns      # the lookup's own region
    printed = capsys.readouterr().out
    assert "50 of 60 left rows matched" in printed
    assert "'los angeles'" in printed                 # a real, unfixable-by-trim miss


# ----------------------------------------------------------------- CONSOLE OUTPUT
class _FakeStream:
    """A stdout stand-in whose tty-ness we control."""

    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


def test_ansi_codes_are_dropped_when_output_is_not_a_terminal(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert core._resolve_colors(_FakeStream(False)) == ("", "", "")


def test_ansi_codes_survive_on_a_real_terminal(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    bold, under, reset = core._resolve_colors(_FakeStream(True))
    assert bold and under and reset


def test_no_color_beats_a_terminal(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert core._resolve_colors(_FakeStream(True)) == ("", "", "")


def test_force_color_beats_a_pipe(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    assert core._resolve_colors(_FakeStream(False)) != ("", "", "")


def test_force_color_zero_turns_colour_off(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "0")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert core._resolve_colors(_FakeStream(True)) == ("", "", "")


def test_no_printer_hardcodes_an_escape_code(capsys, monkeypatch):
    # With the codes resolved to "" (piped output), nothing may still emit an
    # escape: every heading has to go through the module-level constants.
    for name in ("BOLD", "UNDER", "RESET"):
        monkeypatch.setattr(core, name, "")
    Data(SAMPLE).report().nulls().describe().clean().log()
    assert "\033[" not in capsys.readouterr().out


# ----------------------------------------------------------------- NOTEBOOK REPR
def test_repr_html_shows_the_shape_and_the_rows():
    html = Data(SAMPLE)._repr_html_()
    assert "60 rows" in html and "4 cols" in html
    assert "<table" in html


def test_repr_html_says_when_it_truncated():
    assert "showing the first 10" in Data(SAMPLE)._repr_html_()
    short = Data(Data(SAMPLE).df.head(3))
    assert "showing the first" not in short._repr_html_()


def test_repr_html_leaves_repr_and_str_alone():
    d = Data(SAMPLE)
    assert repr(d).startswith("dclean.Data(60x4")
    assert "<table" not in repr(d)
    assert "<table" not in str(d)


# ----------------------------------------------------------------- EXPORT PARITY
def test_to_csv_returns_the_same_object(tmp_path):
    d = Data(SAMPLE)
    out = tmp_path / "out.csv"
    assert d.to_csv(str(out)) is d
    assert out.exists()


def test_to_excel_round_trips(tmp_path):
    pytest.importorskip("openpyxl")
    d = Data(SAMPLE)
    out = tmp_path / "out.xlsx"
    assert d.to_excel(str(out)) is d
    assert Data(str(out)).df.shape == d.df.shape


def test_to_json_round_trips(tmp_path):
    d = Data(SAMPLE)
    out = tmp_path / "out.json"
    assert d.to_json(str(out)) is d
    assert Data(str(out)).df.shape == d.df.shape


def test_to_json_writes_readable_dates(tmp_path):
    d = Data.from_records([{"when": "2025-01-31", "n": 1}]).clean(verbose=False)
    out = tmp_path / "dates.json"
    d.to_json(str(out))
    assert "2025-01-31" in out.read_text()


def test_to_parquet_round_trips(tmp_path):
    pytest.importorskip("pyarrow")
    d = Data(SAMPLE)
    out = tmp_path / "out.parquet"
    assert d.to_parquet(str(out)) is d
    assert Data(str(out)).df.shape == d.df.shape


def test_missing_optional_dependency_names_the_pip_command(monkeypatch):
    def boom(name):
        raise ImportError(name)
    monkeypatch.setattr(core.importlib, "import_module", boom)
    with pytest.raises(ImportError) as e:
        Data(SAMPLE).to_excel("nope.xlsx")
    assert "pip install openpyxl" in str(e.value)
    with pytest.raises(ImportError) as e:
        Data(SAMPLE).to_parquet("nope.parquet")
    assert "pip install pyarrow" in str(e.value)


def test_exports_write_no_index_column(tmp_path):
    d = Data(SAMPLE).filter("age > 18")
    out = tmp_path / "out.csv"
    d.to_csv(str(out))
    assert Data(str(out)).df.columns.tolist() == d.df.columns.tolist()


# ----------------------------------------------------------------- HTML REPORT
def test_report_to_html_writes_a_file_and_returns_self(tmp_path):
    out = tmp_path / "profile.html"
    d = Data(SAMPLE)
    assert d.report(to=str(out)) is d
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "</html>" in html


def test_html_report_is_self_contained(tmp_path):
    # One file that opens anywhere: styles inline, nothing fetched over the
    # network, or it breaks behind a firewall and as an email attachment.
    out = tmp_path / "profile.html"
    Data(SAMPLE).report(to=str(out))
    html = out.read_text(encoding="utf-8")
    assert "<style>" in html
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert "cdn" not in html.lower()


def test_html_report_carries_the_same_facts_as_the_terminal(tmp_path, capsys):
    d = Data("sample_sales.csv")
    d.report()
    printed = capsys.readouterr().out
    out = tmp_path / "profile.html"
    d.report(to=str(out))
    html = out.read_text(encoding="utf-8")

    profile = d._profile()
    for col in d.df.columns:                      # dtypes / null / unique rows
        assert str(col).strip() in html           # to_html trims cell padding
        assert str(col).strip() in printed
    assert str(profile["dupes"]) in html          # duplicate count
    assert "duplicate rows" in printed
    assert "Numeric summary" in html              # numeric stats
    assert "numeric summary" in printed
    for w in profile["warnings"]:                 # warnings
        assert w in html or w.replace("'", "&#x27;") in html


def test_html_report_honours_examples_false(tmp_path):
    d = Data.from_records([{"email": "alice@example.com", "n": 1},
                           {"email": "bob@example.com", "n": 2}])
    shared = tmp_path / "shared.html"
    d.report(examples=False, to=str(shared))
    text = shared.read_text(encoding="utf-8")
    assert "alice@example.com" not in text
    assert "&lt;str&gt;" in text or "<str>" in text
    assert "withheld" in text

    # the default still shows a real example - that is the useful behaviour
    full = tmp_path / "full.html"
    d.report(to=str(full))
    assert "alice@example.com" in full.read_text(encoding="utf-8")


def test_html_report_escapes_values_and_column_names(tmp_path):
    d = Data.from_records([{"<script>": "<img onerror=x>"},
                           {"<script>": "<img onerror=x>"}])
    out = tmp_path / "nasty.html"
    d.report(to=str(out))
    text = out.read_text(encoding="utf-8")
    # nothing from the data may reach the document as live markup
    assert "<script>" not in text.replace("<style>", "")
    assert "<img onerror" not in text
    assert "&lt;script&gt;" in text


def test_report_without_to_still_prints_and_writes_nothing(tmp_path, capsys):
    before = set(os.listdir(str(tmp_path)))
    Data(SAMPLE).report()
    assert "DATASET REPORT" in capsys.readouterr().out
    assert set(os.listdir(str(tmp_path))) == before


def test_html_report_survives_a_frame_with_no_numeric_columns(tmp_path):
    d = Data.from_records([{"city": "NY"}, {"city": "LA"}])
    out = tmp_path / "text.html"
    d.report(to=str(out))
    text = out.read_text(encoding="utf-8")
    assert "Numeric summary" not in text
    assert "Warnings" in text


def test_html_report_says_so_when_there_is_nothing_to_warn_about(tmp_path):
    # repeated values on purpose: unique-per-row text would warn "looks like an ID"
    d = Data.from_records([{"a": 1, "b": "x"}, {"a": 2, "b": "x"}, {"a": 1, "b": "y"}])
    out = tmp_path / "clean.html"
    d.report(to=str(out))
    assert "No data-quality warnings." in out.read_text(encoding="utf-8")


# ------------------------------------------------------------------------- CLI
# Driven through main(argv) rather than a subprocess: the exit code and the
# streams are the contract, and a subprocess would only test pip's shim.
from dclean import cli


def test_cli_report_prints_a_profile(capsys):
    assert cli.main(["report", SAMPLE]) == 0
    out = capsys.readouterr().out
    assert "DATASET REPORT" in out
    assert "duplicate rows" in out


def test_cli_report_can_hide_real_values(capsys, tmp_path):
    src = tmp_path / "people.csv"
    src.write_text("email\nalice@example.com\nbob@example.com\n")
    assert cli.main(["report", str(src), "--no-examples"]) == 0
    out = capsys.readouterr().out
    assert "alice@example.com" not in out
    assert "<str>" in out


def test_cli_report_writes_html(tmp_path, capsys):
    out = tmp_path / "profile.html"
    assert cli.main(["report", SAMPLE, "--html", str(out)]) == 0
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "DATASET REPORT" not in capsys.readouterr().out  # written, not printed


def test_cli_report_html_honours_no_examples(tmp_path):
    src = tmp_path / "people.csv"
    src.write_text("email\nalice@example.com\nbob@example.com\n")
    out = tmp_path / "profile.html"
    assert cli.main(["report", str(src), "--no-examples", "--html", str(out)]) == 0
    assert "alice@example.com" not in out.read_text(encoding="utf-8")


def test_cli_clean_writes_the_cleaned_file(tmp_path, capsys):
    out = tmp_path / "clean.csv"
    assert cli.main(["clean", "sample_sales.csv", "-o", str(out)]) == 0
    assert "CLEAN" in capsys.readouterr().out
    cleaned = Data(str(out))
    assert "order_id" in cleaned.df.columns          # names normalized
    assert not cleaned.df.duplicated().any()         # duplicates gone


def test_cli_clean_nulls_strategy(tmp_path):
    out = tmp_path / "clean.csv"
    assert cli.main(["clean", "sample_sales.csv", "--nulls", "drop",
                     "-o", str(out), "-q"]) == 0
    assert Data(str(out)).df.isna().sum().sum() == 0


def test_cli_clean_quiet_prints_nothing(capsys, tmp_path):
    out = tmp_path / "clean.csv"
    assert cli.main(["clean", "sample_sales.csv", "-o", str(out), "-q"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_clean_no_dates_leaves_dates_as_text(tmp_path):
    out = tmp_path / "clean.csv"
    assert cli.main(["clean", "sample_sales.csv", "--no-dates",
                     "-o", str(out), "-q"]) == 0
    # order_date stays text, so it round-trips through csv as an object column
    assert not pd.api.types.is_datetime64_any_dtype(Data(str(out)).df["order_date"])


def test_cli_clean_writes_every_format_it_can_read(tmp_path):
    pytest.importorskip("pyarrow")
    out = tmp_path / "clean.parquet"
    assert cli.main(["clean", "sample_sales.csv", "-o", str(out), "-q"]) == 0
    assert Data(str(out)).df.shape[0] > 0


def test_cli_clean_rejects_an_output_it_cannot_write(tmp_path, capsys):
    out = tmp_path / "clean.txt"
    assert cli.main(["clean", "sample_sales.csv", "-o", str(out), "-q"]) == 1
    err = capsys.readouterr().err
    assert "don't know how to write" in err
    assert not out.exists()


def test_cli_head_shows_n_rows(capsys):
    assert cli.main(["head", SAMPLE, "-n", "3"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") <= 6          # header + separator + 3 rows
    for col in Data(SAMPLE).df.columns:
        assert col in out


def test_cli_head_defaults_to_five_rows(capsys):
    assert cli.main(["head", SAMPLE]) == 0
    assert capsys.readouterr().out.strip().count("\n") == 6


def test_cli_missing_file_is_one_line_not_a_traceback(capsys):
    assert cli.main(["report", "definitely_not_here.csv"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "dcleaner: no such file: definitely_not_here.csv"
    assert "Traceback" not in captured.err


def test_cli_unsupported_file_type_is_one_line_not_a_traceback(tmp_path, capsys):
    junk = tmp_path / "notes.txt"
    junk.write_text("hello")
    assert cli.main(["head", str(junk)]) == 1
    captured = capsys.readouterr()
    assert captured.err.count("\n") == 1
    assert "Unsupported file type" in captured.err
    assert "Traceback" not in captured.err


def test_cli_version_flag_matches_the_package(capsys):
    import dclean
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert dclean.__version__ in capsys.readouterr().out


def test_cli_with_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    for command in ("report", "clean", "head"):
        assert command in out


def test_cli_rejects_an_unknown_nulls_strategy(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["clean", "sample_sales.csv", "--nulls", "nonsense"])
    assert e.value.code == 2                       # argparse's own usage error
    assert "invalid choice" in capsys.readouterr().err


def test_cli_entry_point_is_declared():
    # the [project.scripts] name is what `pip install dcleaner` puts on PATH
    root = os.path.dirname(HERE)
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        pyproject = fh.read()
    assert '[project.scripts]' in pyproject
    assert 'dcleaner = "dclean.cli:main"' in pyproject


# --------------------------------------------------------------------- RECIPES
import json


def test_recipe_round_trips_a_pipeline(tmp_path):
    jan = tmp_path / "jan.csv"
    jan.write_text("Units,City\n10,NY\n3,LA\n8,SF\n")
    feb = tmp_path / "feb.csv"
    feb.write_text("Units,City\n20,NY\n1,LA\n6,SF\n")
    recipe = tmp_path / "monthly.json"

    (Data(str(jan)).clean(verbose=False).filter("units > 5")
        .save_recipe(str(recipe)))
    out = Data(str(feb)).apply_recipe(str(recipe), verbose=False)

    assert out.df["units"].tolist() == [20, 6]      # the filter really ran
    assert "units" in out.df.columns                # clean() normalized names


def test_recipe_file_matches_the_documented_schema(tmp_path):
    recipe = tmp_path / "r.json"
    Data(SAMPLE).dropna().filter("age > 18").save_recipe(str(recipe))
    doc = json.loads(recipe.read_text())

    assert doc["recipe"] == core.RECIPE_FORMAT
    assert doc["dcleaner"] == __import__("dclean").__version__
    assert "created" in doc
    assert [s["method"] for s in doc["steps"]] == ["dropna", "filter"]
    for step in doc["steps"]:
        assert set(step) == {"method", "args", "kwargs", "display"}
    assert doc["steps"][1]["args"] == ["age > 18"]


def test_save_recipe_returns_self_and_apply_returns_a_new_object(tmp_path):
    recipe = tmp_path / "r.json"
    d = Data(SAMPLE).dropna()
    assert d.save_recipe(str(recipe)) is d           # output method
    replayed = Data(SAMPLE).apply_recipe(str(recipe), verbose=False)
    assert replayed is not d                         # transform
    assert isinstance(replayed, Data)


def test_replayed_steps_show_up_in_the_new_log(tmp_path):
    recipe = tmp_path / "r.json"
    Data(SAMPLE).dropna().filter("age > 18").save_recipe(str(recipe))
    out = Data(SAMPLE).apply_recipe(str(recipe), verbose=False)
    assert out.steps() == ["dropna()", "filter('age > 18')"]


def test_steps_still_returns_plain_strings():
    d = Data(SAMPLE).dropna().filter("age > 18")
    assert d.steps() == ["dropna()", "filter('age > 18')"]
    assert all(isinstance(s, str) for s in d.steps())


def test_display_string_is_recorded_but_never_parsed(tmp_path):
    # replay reads the structure; an edited display line must change nothing
    recipe = tmp_path / "r.json"
    Data(SAMPLE).filter("age > 18").save_recipe(str(recipe))
    doc = json.loads(recipe.read_text())
    doc["steps"][0]["display"] = "filter('age > 999')  # a lie"
    recipe.write_text(json.dumps(doc))
    out = Data(SAMPLE).apply_recipe(str(recipe), verbose=False)
    assert len(out) == len(Data(SAMPLE).filter("age > 18"))


def test_recipe_survives_names_that_would_break_a_parser(tmp_path):
    # the reason steps are structured: this column name has a quote and a bracket
    nasty = "it's (a) column"
    src = tmp_path / "nasty.csv"
    pd.DataFrame({nasty: [1, 2, 3], "b": [1, 2, 3]}).to_csv(src, index=False)
    recipe = tmp_path / "r.json"
    Data(str(src)).drop([nasty]).save_recipe(str(recipe))
    out = Data(str(src)).apply_recipe(str(recipe), verbose=False)
    assert nasty not in out.df.columns
    assert "b" in out.df.columns


def test_save_recipe_refuses_a_join_instead_of_dropping_it(tmp_path):
    left = tmp_path / "l.csv"
    left.write_text("city,n\nNY,1\n")
    right = tmp_path / "r.csv"
    right.write_text("city,region\nNY,East\n")
    d = Data(str(left)).join(str(right), on="city", verbose=False)
    with pytest.raises(ValueError) as e:
        d.save_recipe(str(tmp_path / "r.json"))
    assert "cannot be replayed" in str(e.value)
    assert "join" in str(e.value)


def test_save_recipe_points_groupby_at_the_one_call_form(tmp_path):
    d = Data(SAMPLE).dropna().groupby("city").agg("mean", "salary")
    with pytest.raises(ValueError) as e:
        d.save_recipe(str(tmp_path / "r.json"))
    assert "mean(col, by=...)" in str(e.value)


def test_the_one_call_aggregate_does_record(tmp_path):
    recipe = tmp_path / "r.json"
    Data(SAMPLE).dropna().mean("salary", by="city").save_recipe(str(recipe))
    out = Data(SAMPLE).dropna().apply_recipe(str(recipe), verbose=False)
    assert len(out) == 3                              # NY, LA, SF


def test_apply_recipe_refuses_a_method_that_is_not_whitelisted(tmp_path):
    recipe = tmp_path / "evil.json"
    recipe.write_text(json.dumps({
        "recipe": core.RECIPE_FORMAT,
        "steps": [{"method": "to_csv", "args": ["/tmp/pwned.csv"],
                   "kwargs": {}, "display": "to_csv(...)"}],
    }))
    with pytest.raises(ValueError) as e:
        Data(SAMPLE).apply_recipe(str(recipe))
    assert "will not replay" in str(e.value)
    assert not os.path.exists("/tmp/pwned.csv")


def test_apply_recipe_refuses_an_unknown_format(tmp_path):
    recipe = tmp_path / "future.json"
    recipe.write_text(json.dumps({"recipe": 99, "steps": []}))
    with pytest.raises(ValueError) as e:
        Data(SAMPLE).apply_recipe(str(recipe))
    assert "recipe format" in str(e.value)


def test_apply_recipe_refuses_a_file_that_is_not_a_recipe(tmp_path):
    recipe = tmp_path / "notes.json"
    recipe.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError) as e:
        Data(SAMPLE).apply_recipe(str(recipe))
    assert "not a dclean recipe" in str(e.value)


def test_apply_recipe_says_which_step_failed(tmp_path):
    recipe = tmp_path / "r.json"
    Data(SAMPLE).dropna().filter("age > 18").save_recipe(str(recipe))
    other = tmp_path / "other.csv"
    other.write_text("height\n180\n")             # no 'age' column
    with pytest.raises(ValueError) as e:
        Data(str(other)).apply_recipe(str(recipe))
    message = str(e.value)
    assert "step 2" in message
    assert "filter" in message


def test_save_recipe_refuses_an_argument_that_is_not_json(tmp_path):
    d = Data(SAMPLE).fillna(object())
    with pytest.raises(ValueError) as e:
        d.save_recipe(str(tmp_path / "r.json"))
    assert "not JSON" in str(e.value)


def test_recipe_replays_a_long_chain_faithfully(tmp_path):
    src = tmp_path / "src.csv"
    src.write_text(" Total Sales ,City,Units\n$1{sep}200.50,NY,10\n"
                   "$300.00, la ,2\n$1{sep}200.50,NY,10\n".format(sep=","))
    recipe = tmp_path / "r.json"
    built = (Data(str(src)).clean(verbose=False)
             .filter("units > 5")
             .mutate(double="units * 2")
             .sort("units", ascending=False)
             .select("city", "units", "double"))
    built.save_recipe(str(recipe))
    replayed = Data(str(src)).apply_recipe(str(recipe), verbose=False)
    pd.testing.assert_frame_equal(built.df.reset_index(drop=True),
                                  replayed.df.reset_index(drop=True))


def test_apply_recipe_announces_itself_unless_quiet(tmp_path, capsys):
    recipe = tmp_path / "r.json"
    Data(SAMPLE).dropna().save_recipe(str(recipe))
    capsys.readouterr()
    Data(SAMPLE).apply_recipe(str(recipe))
    assert "replayed 1 steps" in capsys.readouterr().out
    Data(SAMPLE).apply_recipe(str(recipe), verbose=False)
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------- TYPING
def test_py_typed_marker_ships_with_the_package():
    import dclean
    marker = os.path.join(os.path.dirname(dclean.__file__), "py.typed")
    assert os.path.exists(marker), "py.typed is what tells a checker to read our hints"


def test_py_typed_is_declared_as_package_data():
    root = os.path.dirname(HERE)
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        pyproject = fh.read()
    # without this the marker exists in the repo but never reaches the wheel
    assert 'dclean = ["py.typed"]' in pyproject


def test_public_methods_carry_type_hints():
    import inspect
    skip = {"samples"}                     # staticmethod with no self to bind
    missing = []
    for name in dir(Data):
        if name.startswith("_") or name in skip:
            continue
        member = inspect.getattr_static(Data, name)
        func = member.__func__ if isinstance(member, (staticmethod, classmethod)) \
            else getattr(member, "func", member)
        if not inspect.isfunction(func):
            continue
        if inspect.signature(func).return_annotation is inspect.Signature.empty:
            missing.append(name)
    assert not missing, "public methods with no return annotation: %s" % missing


def test_the_version_is_the_same_in_both_places():
    import dclean
    root = os.path.dirname(HERE)
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        pyproject = fh.read()
    assert 'version = "%s"' % dclean.__version__ in pyproject


# ------------------------------------------------------------- PLOT COLOURS
def test_plots_use_the_colourblind_safe_cycle():
    d = Data.from_records([{"city": "NY", "n": 3}, {"city": "LA", "n": 1},
                           {"city": "SF", "n": 4}])
    line = d.plot("line", x="city", y="n", show=False).to_fig()
    assert line.axes[0].lines[0].get_color() == core.OKABE_ITO[0]

    # pandas reads the cycle from rcParams when it draws a bar, so this is the
    # case an axes-level prop_cycle would silently miss
    bar = d.plot("bar", x="city", y="n", show=False).to_fig()
    facecolor = bar.axes[0].patches[0].get_facecolor()
    assert matplotlib.colors.to_hex(facecolor).upper() == core.OKABE_ITO[0]


def test_a_second_series_gets_the_next_okabe_ito_colour():
    d = Data.from_records([{"city": "NY", "n": 3, "m": 5},
                           {"city": "LA", "n": 1, "m": 2}])
    fig = d.plot("line", x="city", show=False).to_fig()
    colours = [line.get_color() for line in fig.axes[0].lines]
    assert colours == list(core.OKABE_ITO[:2])


def test_an_explicit_colour_beats_the_default():
    d = Data.from_records([{"city": "NY", "n": 3}, {"city": "LA", "n": 1}])
    fig = d.plot("line", x="city", y="n", color="red", show=False).to_fig()
    assert fig.axes[0].lines[0].get_color() == "red"


def test_the_palette_does_not_leak_into_global_state():
    before = matplotlib.rcParams["axes.prop_cycle"]
    Data.from_records([{"a": 1}, {"a": 2}]).plot("line", y="a", show=False)
    assert matplotlib.rcParams["axes.prop_cycle"] == before


def test_plot_corr_uses_a_diverging_map_anchored_at_zero():
    fig = Data(SAMPLE).dropna().plot_corr(show=False).to_fig()
    image = fig.axes[0].images[0]
    assert image.get_cmap().name == core.DIVERGING
    assert image.get_clim() == (-1.0, 1.0)   # midpoint really is "no correlation"


def test_plot_corr_cmap_is_still_overridable():
    fig = Data(SAMPLE).dropna().plot_corr(cmap="viridis", show=False).to_fig()
    assert fig.axes[0].images[0].get_cmap().name == "viridis"


def test_the_palette_is_the_okabe_ito_set():
    # eight hues, all distinct, all valid colours
    assert len(core.OKABE_ITO) == len(set(core.OKABE_ITO)) == 8
    for colour in core.OKABE_ITO:
        matplotlib.colors.to_rgb(colour)     # raises if it is not a colour
