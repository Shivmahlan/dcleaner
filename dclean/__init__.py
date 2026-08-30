"""dclean - fluent data cleaning & visualization for pandas users.

Two ways in:

    from dclean import Data          # the fluent, chainable object
    import dclean; dclean.clean(...) # one-call helpers ("just tell it")
"""
from typing import Any, Optional

from .core import Data


def load(source: Any) -> Data:
    """Load any supported file into a ``Data``. ``dclean.load("sales.csv")``"""
    return Data(source)


def clean(source: Any, to: Optional[str] = None, nulls: str = "keep",
          verbose: bool = True) -> Data:
    """Load a file, auto-clean it, and optionally write it back out.

    The whole job in one call - you name the file, the library works out the
    column names, types, dates, blanks and duplicates::

        import dclean
        dclean.clean("messy.csv", to="clean.csv")

    ``nulls`` is passed through to :meth:`Data.clean` ("keep"/"drop"/"fill").
    Returns the cleaned ``Data`` so you can keep chaining.
    """
    d = Data(source).clean(nulls=nulls, verbose=verbose)
    if to:
        d = d.to_csv(to)
    return d


def report(source: Any) -> Data:
    """Load a file and print a full data-quality profile of it.

        import dclean
        dclean.report("sales.csv")

    Returns the loaded ``Data``.
    """
    return Data(source).report()


__version__ = "0.3.0"
__all__ = ["Data", "load", "clean", "report", "__version__"]
