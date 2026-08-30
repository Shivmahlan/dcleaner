"""Command-line front end for dclean.

Every subcommand is a thin wrapper over the same API you would use from Python
- ``dclean.clean``, ``dclean.report``, ``Data`` - so the terminal and the
library can never disagree about what "clean" means::

    dcleaner report sales.csv
    dcleaner report sales.csv --no-examples --html profile.html
    dcleaner clean messy.csv -o clean.csv --nulls fill
    dcleaner head sales.csv -n 20

Errors a user can actually cause - a missing file, a file type dclean does not
read - exit 1 with one line on stderr. A traceback is for a bug in dclean, not
for a typo in a filename.
"""
import argparse
import contextlib
import io
import sys
from typing import List, Optional

from . import __version__
from .core import Data


class CliError(Exception):
    """A problem the user can fix: exits 1 with one line, never a traceback."""


def _load(path: str) -> Data:
    """Load ``path`` into a ``Data``, turning the usual failures into CliError."""
    try:
        return Data(path)
    except FileNotFoundError:
        raise CliError("no such file: {}".format(path))
    except ValueError as e:
        # Data._load() raises this for a suffix it does not read.
        raise CliError(str(e))
    except ImportError as e:
        # An optional reader (parquet, excel) is not installed.
        raise CliError(str(e))


def _build_parser() -> argparse.ArgumentParser:
    """The argparse parser. Stdlib only - no dependency for a convenience."""
    parser = argparse.ArgumentParser(
        prog="dcleaner",
        description="Clean, profile and peek at data files from the shell.",
        epilog="Every subcommand wraps the Python API of the same name.")
    parser.add_argument("--version", action="version",
                        version="dcleaner {}".format(__version__))
    subs = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_report = subs.add_parser(
        "report", help="print a full data-quality profile of a file",
        description="Profile a file: dtypes, nulls, unique counts, duplicate "
                    "rows, numeric stats and data-quality warnings.")
    p_report.add_argument("file", metavar="FILE", help="the data file to profile")
    p_report.add_argument(
        "--no-examples", action="store_true",
        help="show each column's TYPE instead of a real value from your data "
             "- use this for anything you are going to share")
    p_report.add_argument(
        "--html", metavar="OUT",
        help="write the profile to a self-contained HTML file instead of "
             "printing it")

    p_clean = subs.add_parser(
        "clean", help="auto-clean a file and optionally write it out",
        description="Normalize column names, trim text, coerce numbers and "
                    "dates, drop empty rows/columns and duplicates.")
    p_clean.add_argument("file", metavar="FILE", help="the data file to clean")
    p_clean.add_argument("-o", "--out", metavar="OUT",
                         help="write the cleaned data here (csv/xlsx/json/parquet)")
    p_clean.add_argument("--nulls", choices=("keep", "drop", "fill"),
                         default="keep",
                         help="what to do with missing values (default: keep)")
    p_clean.add_argument("--no-dates", action="store_true",
                         help="do not try to parse date-looking text columns")
    p_clean.add_argument("-q", "--quiet", action="store_true",
                         help="clean silently - print nothing")

    p_head = subs.add_parser(
        "head", help="print the first rows of a file",
        description="Peek at a file without opening a Python session.")
    p_head.add_argument("file", metavar="FILE", help="the data file to peek at")
    p_head.add_argument("-n", type=int, default=5, metavar="N",
                        help="how many rows to show (default: 5)")

    return parser


def _write(data: Data, path: str) -> Data:
    """Write ``data`` to ``path``, picking the writer from the suffix."""
    lowered = str(path).lower()
    if lowered.endswith((".csv", ".csv.gz")):
        return data.to_csv(path)
    if lowered.endswith((".xls", ".xlsx")):
        return data.to_excel(path)
    if lowered.endswith(".json"):
        return data.to_json(path)
    if lowered.endswith(".parquet"):
        return data.to_parquet(path)
    raise CliError(
        "don't know how to write {!r} - use .csv, .xlsx, .json or "
        ".parquet".format(path))


def _cmd_report(args: argparse.Namespace) -> int:
    d = _load(args.file)
    d.report(examples=not args.no_examples, to=args.html)
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    d = _load(args.file).clean(nulls=args.nulls, dates=not args.no_dates,
                               verbose=not args.quiet)
    if args.out:
        # -q means print nothing, and the writers announce where they wrote.
        hush = contextlib.redirect_stdout(io.StringIO()) if args.quiet \
            else contextlib.nullcontext()
        try:
            with hush:
                _write(d, args.out)
        except ImportError as e:      # optional writer not installed
            raise CliError(str(e))
    return 0


def _cmd_head(args: argparse.Namespace) -> int:
    _load(args.file).head(args.n)
    return 0


_COMMANDS = {"report": _cmd_report, "clean": _cmd_clean, "head": _cmd_head}


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI. Returns the exit code: 0 on success, 1 on a handled error.

    Call it directly to test the CLI - ``main(["report", "sales.csv"])`` - with
    no subprocess in the way.
    """
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if not args.command:
        parser.print_help()
        return 0
    try:
        return _COMMANDS[args.command](args)
    except CliError as e:
        sys.stderr.write("dcleaner: {}\n".format(e))
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        sys.stderr.write("dcleaner: interrupted\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised via the entry point
    sys.exit(main())
