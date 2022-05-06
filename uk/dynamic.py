"""
Resolve functions by their fully qualified name (setuptools style).

>>> import os.path
>>> import_function('os.path:isdir') is os.path.isdir
True
"""

import importlib
from typing import Callable


def import_function(fully_qualified_name: str) -> Callable:
    package, name = fully_qualified_name.rsplit(':', maxsplit=1)
    return getattr(importlib.import_module(package), name)


if __name__ == '__main__':
    import sys

    for name in sys.argv[1:]:
        print(name, import_function(name).__doc__)
    else:
        import doctest
        doctest.testmod(verbose=True)