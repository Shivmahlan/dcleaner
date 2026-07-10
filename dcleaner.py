"""dcleaner - fluent data cleaning & visualization for pandas users.

The importable package is named ``dclean`` (so ``from dclean import Data``
works), but the distribution on PyPI is named ``dcleaner``. This shim lets
you also do::

    import dcleaner
    dcleaner.Data("file.csv").dropna().head()

Both forms resolve to the same ``Data`` class.
"""
from dclean import Data, __version__

__all__ = ["Data", "__version__"]
