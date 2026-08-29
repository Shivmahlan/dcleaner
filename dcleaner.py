"""dcleaner - fluent data cleaning & visualization for pandas users.

The importable package is named ``dclean`` (so ``from dclean import Data``
works), but the distribution on PyPI is named ``dcleaner``. This shim lets
you also do::

    import dcleaner
    dcleaner.clean("messy.csv", to="clean.csv")
    dcleaner.Data("file.csv").dropna().head()

Both forms resolve to the same objects.
"""
from dclean import Data, load, clean, report, __version__

__all__ = ["Data", "load", "clean", "report", "__version__"]
